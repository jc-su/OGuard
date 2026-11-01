import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=[
        "out_qty", "out_base_price", "out_disc_price", 
        "out_charge", "out_disc", "out_count"
    ]
)
@triton.jit
def q1_kernel(
    rf_ptr, ls_ptr, qty_ptr, price_ptr, disc_ptr, tax_ptr, shipdate_ptr,
    shipdate_hi, num_ls, n_elements,
    out_qty, out_base_price, out_disc_price, out_charge, out_disc, out_count,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate early to quickly evaluate the global filter
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi + 1)
    valid = mask & (shipdate <= shipdate_hi)
    
    # Load category keys
    rf = tl.load(rf_ptr + offs, mask=valid, other=0)
    ls = tl.load(ls_ptr + offs, mask=valid, other=0)
    valid = valid & (rf >= 0) & (ls >= 0)
    
    group_id = rf * num_ls + ls
    
    # Load measure columns mapped strictly to valid lanes
    qty = tl.load(qty_ptr + offs, mask=valid, other=0.0)
    price = tl.load(price_ptr + offs, mask=valid, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=valid, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=valid, other=0.0)
    
    # Derived expressions
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)
    
    zero_f64 = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float64)
    one_i64 = tl.full([BLOCK_SIZE], 1, dtype=tl.int64)
    zero_i64 = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
    
    # Block-local reduction: highly performant since NUM_GROUPS is tiny (typically ~4-6)
    # Loop unrolled by Triton, eliminating large amounts of atomic contention on global memory.
    for g in tl.static_range(0, NUM_GROUPS):
        g_mask = valid & (group_id == g)
        
        sum_qty = tl.sum(tl.where(g_mask, qty, zero_f64))
        sum_price = tl.sum(tl.where(g_mask, price, zero_f64))
        sum_disc_price = tl.sum(tl.where(g_mask, disc_price, zero_f64))
        sum_charge = tl.sum(tl.where(g_mask, charge, zero_f64))
        sum_disc = tl.sum(tl.where(g_mask, disc, zero_f64))
        sum_count = tl.sum(tl.where(g_mask, one_i64, zero_i64))
        
        # Only write to global memory if the block has data for this group
        if sum_count > 0:
            tl.atomic_add(out_qty + g, sum_qty)
            tl.atomic_add(out_base_price + g, sum_price)
            tl.atomic_add(out_disc_price + g, sum_disc_price)
            tl.atomic_add(out_charge + g, sum_charge)
            tl.atomic_add(out_disc + g, sum_disc)
            tl.atomic_add(out_count + g, sum_count)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q1_triton_")
    path = os.path.join(tmpdir, "_q1_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q1_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
         
    Description:
    Triton fused pass over `lineitem` to execute filtering, derived metrics calculation, and aggregation.
    Uses categorical mapping for grouping combinations, which reduces grouping space to fewer than 10 groups,
    allowing an ultra-fast block-level `tl.static_range` reduction before writing to global memory.
    
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]

    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Convert string columns to small categorical codes
    rf_codes = df_lineitem["l_returnflag"].astype("category")
    ls_codes = df_lineitem["l_linestatus"].astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    num_rf = len(rf_categories)
    num_ls = len(ls_categories)
    num_groups = num_rf * num_ls

    l_returnflag_code_t = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy())
    l_linestatus_code_t = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy())

    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    n = l_quantity_t.numel()
    device = l_quantity_t.device
    
    # Pre-allocate output accumulators
    out_qty = torch.zeros(num_groups, dtype=torch.float64, device=device)
    out_base_price = torch.zeros(num_groups, dtype=torch.float64, device=device)
    out_disc_price = torch.zeros(num_groups, dtype=torch.float64, device=device)
    out_charge = torch.zeros(num_groups, dtype=torch.float64, device=device)
    out_disc = torch.zeros(num_groups, dtype=torch.float64, device=device)
    out_count = torch.zeros(num_groups, dtype=torch.int64, device=device)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K.q1_kernel[grid](
        l_returnflag_code_t, l_linestatus_code_t, l_quantity_t, l_extendedprice_t,
        l_discount_t, l_tax_t, l_shipdate_days_t, shipdate_hi_days, num_ls, n,
        out_qty, out_base_price, out_disc_price, out_charge, out_disc, out_count,
        NUM_GROUPS=num_groups
    )

    # Filter out empty group buckets and compute averages
    valid_mask = out_count > 0
    valid_indices = torch.nonzero(valid_mask).squeeze(1)

    rf_out_idx = valid_indices // num_ls
    ls_out_idx = valid_indices % num_ls

    sum_qty_out = out_qty[valid_indices]
    sum_base_price_out = out_base_price[valid_indices]
    sum_disc_price_out = out_disc_price[valid_indices]
    sum_charge_out = out_charge[valid_indices]
    sum_disc_out = out_disc[valid_indices]
    count_out = out_count[valid_indices]

    count_f = count_out.to(torch.float64)
    avg_qty_out = sum_qty_out / count_f
    avg_price_out = sum_base_price_out / count_f
    avg_disc_out = sum_disc_out / count_f

    # Decode categories back into expected literal strings (already sorted by keys)
    rf_strs = [rf_categories[i] for i in rf_out_idx.cpu().tolist()]
    ls_strs = [ls_categories[i] for i in ls_out_idx.cpu().tolist()]

    return cudf.DataFrame({
        "l_returnflag": cudf.Series(rf_strs),
        "l_linestatus": cudf.Series(ls_strs),
        "sum_qty": sum_qty_out,
        "sum_base_price": sum_base_price_out,
        "sum_disc_price": sum_disc_price_out,
        "sum_charge": sum_charge_out,
        "avg_qty": avg_qty_out,
        "avg_price": avg_price_out,
        "avg_disc": avg_disc_out,
        "count_order": count_out,
    })
