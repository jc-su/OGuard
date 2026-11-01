import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton
import triton.language as tl

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [2048, 4096, 8192]
        for w in [8, 16]
        for s in [3, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q1_kernel(
    rf_ptr, ls_ptr, shipdate_ptr,
    qty_ptr, price_ptr, disc_ptr, tax_ptr,
    out_ptr,
    shipdate_limit,
    n_ls,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all columns for this block
    rf = tl.load(rf_ptr + offs, mask=mask, other=0)
    ls = tl.load(ls_ptr + offs, mask=mask, other=0)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_limit + 1)
    
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=mask, other=0.0)

    # Date filter
    date_ok = shipdate <= shipdate_limit
    valid_mask = mask & date_ok
    
    # Compute group index for each element: rf * n_ls + ls
    group_idx = rf * n_ls + ls
    
    # Compute derived values
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)
    
    # Mask values for valid rows only
    qty_val = tl.where(valid_mask, qty, 0.0)
    price_val = tl.where(valid_mask, price, 0.0)
    disc_price_val = tl.where(valid_mask, disc_price, 0.0)
    charge_val = tl.where(valid_mask, charge, 0.0)
    disc_val = tl.where(valid_mask, disc, 0.0)
    count_val = valid_mask.to(tl.float64)
    
    # For each of the 6 groups, accumulate using tl.where + tl.sum
    # Then atomically add to global output
    # Group 0
    g0_mask = (group_idx == 0)
    tl.atomic_add(out_ptr + 0, tl.sum(tl.where(g0_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 1, tl.sum(tl.where(g0_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 2, tl.sum(tl.where(g0_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 3, tl.sum(tl.where(g0_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 4, tl.sum(tl.where(g0_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 5, tl.sum(tl.where(g0_mask, count_val, 0.0)))
    
    # Group 1
    g1_mask = (group_idx == 1)
    tl.atomic_add(out_ptr + 6, tl.sum(tl.where(g1_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 7, tl.sum(tl.where(g1_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 8, tl.sum(tl.where(g1_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 9, tl.sum(tl.where(g1_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 10, tl.sum(tl.where(g1_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 11, tl.sum(tl.where(g1_mask, count_val, 0.0)))
    
    # Group 2
    g2_mask = (group_idx == 2)
    tl.atomic_add(out_ptr + 12, tl.sum(tl.where(g2_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 13, tl.sum(tl.where(g2_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 14, tl.sum(tl.where(g2_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 15, tl.sum(tl.where(g2_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 16, tl.sum(tl.where(g2_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 17, tl.sum(tl.where(g2_mask, count_val, 0.0)))
    
    # Group 3
    g3_mask = (group_idx == 3)
    tl.atomic_add(out_ptr + 18, tl.sum(tl.where(g3_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 19, tl.sum(tl.where(g3_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 20, tl.sum(tl.where(g3_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 21, tl.sum(tl.where(g3_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 22, tl.sum(tl.where(g3_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 23, tl.sum(tl.where(g3_mask, count_val, 0.0)))
    
    # Group 4
    g4_mask = (group_idx == 4)
    tl.atomic_add(out_ptr + 24, tl.sum(tl.where(g4_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 25, tl.sum(tl.where(g4_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 26, tl.sum(tl.where(g4_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 27, tl.sum(tl.where(g4_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 28, tl.sum(tl.where(g4_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 29, tl.sum(tl.where(g4_mask, count_val, 0.0)))
    
    # Group 5
    g5_mask = (group_idx == 5)
    tl.atomic_add(out_ptr + 30, tl.sum(tl.where(g5_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 31, tl.sum(tl.where(g5_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 32, tl.sum(tl.where(g5_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 33, tl.sum(tl.where(g5_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 34, tl.sum(tl.where(g5_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 35, tl.sum(tl.where(g5_mask, count_val, 0.0)))
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
    Description: TPC-H Q1 pricing summary report. Uses a single fused Triton kernel that loads all 
                 columns once, filters by date, computes group indices, and accumulates per-group 
                 sums via block-level reduction + atomic adds. This replaces multiple cuDF/PyTorch 
                 passes (mask, index, unique, scatter_add) with one GPU kernel launch.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode categorical columns to int32 codes
    rf_series = df_lineitem["l_returnflag"].astype("category")
    ls_series = df_lineitem["l_linestatus"].astype("category")
    
    rf_categories = rf_series.cat.categories.to_pandas().tolist()
    ls_categories = ls_series.cat.categories.to_pandas().tolist()
    
    n_rf = len(rf_categories)
    n_ls = len(ls_categories)
    n_groups = n_rf * n_ls  # Should be 6 for TPC-H
    
    # Tensors
    l_returnflag_code_t = torch.as_tensor(rf_series.cat.codes.astype("int32").to_cupy(), device="cuda")
    l_linestatus_code_t = torch.as_tensor(ls_series.cat.codes.astype("int32").to_cupy(), device="cuda")
    
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64, device="cuda")
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64, device="cuda")
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64, device="cuda")
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64, device="cuda")
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(), device="cuda"
    )
    
    n_elements = l_quantity_t.numel()
    
    # Output buffer: [n_groups * 6 aggregates]
    # Layout: out[group * 6 + agg] where agg in [sum_qty, sum_base, sum_disc_price, sum_charge, sum_disc, count]
    out_buffer = torch.zeros(n_groups * 6, dtype=torch.float64, device="cuda")
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    _K._q1_kernel[grid](
        l_returnflag_code_t, l_linestatus_code_t, l_shipdate_days_t,
        l_quantity_t, l_extendedprice_t, l_discount_t, l_tax_t,
        out_buffer,
        shipdate_hi_days,
        n_ls,
        n_elements,
    )
    
    # Post-processing on host
    out_tensor = out_buffer.reshape(n_groups, 6)
    out_cpu = out_tensor.cpu()
    
    results = []
    for r_idx in range(n_rf):
        for l_idx in range(n_ls):
            group_idx = r_idx * n_ls + l_idx
            count = out_cpu[group_idx, 5].item()
            if count == 0:
                continue
            
            rf_str = rf_categories[r_idx]
            ls_str = ls_categories[l_idx]
            
            sum_qty = out_cpu[group_idx, 0].item()
            sum_base = out_cpu[group_idx, 1].item()
            sum_disc_price = out_cpu[group_idx, 2].item()
            sum_charge = out_cpu[group_idx, 3].item()
            sum_disc = out_cpu[group_idx, 4].item()
            
            avg_qty = sum_qty / count
            avg_price = sum_base / count
            avg_disc = sum_disc / count
            
            results.append({
                "l_returnflag": rf_str,
                "l_linestatus": ls_str,
                "sum_qty": sum_qty,
                "sum_base_price": sum_base,
                "sum_disc_price": sum_disc_price,
                "sum_charge": sum_charge,
                "avg_qty": avg_qty,
                "avg_price": avg_price,
                "avg_disc": avg_disc,
                "count_order": int(count),
            })
    
    # Sort by l_returnflag, l_linestatus
    results.sort(key=lambda x: (x["l_returnflag"], x["l_linestatus"]))
    
    return cudf.DataFrame(results)
