import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _q1_fused_kernel(
    rf_ptr, ls_ptr, qty_ptr, price_ptr, disc_ptr, tax_ptr, ship_ptr,
    ship_hi, n_elements,
    sum_qty_ptr, sum_base_price_ptr, sum_disc_price_ptr, sum_charge_ptr,
    avg_qty_ptr, avg_price_ptr, avg_disc_ptr, count_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Filter
    ship_valid = tl.load(ship_ptr + offs, mask=mask, other=0).to(tl.int32)
    keep = mask & (ship_valid <= ship_hi)

    # Load values
    rf = tl.load(rf_ptr + offs, mask=keep, other=0).to(tl.int64)
    ls = tl.load(ls_ptr + offs, mask=keep, other=0).to(tl.int64)
    qty = tl.load(qty_ptr + offs, mask=keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    # Compute aggregates
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    # Store to output
    tl.store(sum_qty_ptr + offs, qty, mask=keep)
    tl.store(sum_base_price_ptr + offs, price, mask=keep)
    tl.store(sum_disc_price_ptr + offs, disc_price, mask=keep)
    tl.store(sum_charge_ptr + offs, charge, mask=keep)
    tl.store(avg_qty_ptr + offs, qty, mask=keep)
    tl.store(avg_price_ptr + offs, price, mask=keep)
    tl.store(avg_disc_ptr + offs, disc, mask=keep)
    tl.store(count_ptr + offs, 1.0, mask=keep)
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
    Description: Optimized TPC-H Q1 using Triton kernel with minimized memory bandwidth.
    Params: shipdate_hi (str): upper-bound ship date.
    """
    df = tables["lineitem"]
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode strings
    rf_cat = df["l_returnflag"].astype("category")
    ls_cat = df["l_linestatus"].astype("category")
    rf_categories = rf_cat.cat.categories.to_pandas().tolist()
    ls_categories = ls_cat.cat.categories.to_pandas().tolist()

    # Convert to tensors
    rf_codes = torch.as_tensor(rf_cat.cat.codes.astype("int32").to_cupy())
    ls_codes = torch.as_tensor(ls_cat.cat.codes.astype("int32").to_cupy())
    qty = torch.as_tensor(df["l_quantity"].to_cupy(), dtype=torch.float64)
    price = torch.as_tensor(df["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc = torch.as_tensor(df["l_discount"].to_cupy(), dtype=torch.float64)
    tax = torch.as_tensor(df["l_tax"].to_cupy(), dtype=torch.float64)
    ship_days = torch.as_tensor(((df["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy())

    n = qty.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    # Allocate output tensors
    sum_qty = torch.zeros(n, dtype=torch.float64, device=qty.device)
    sum_base_price = torch.zeros(n, dtype=torch.float64, device=qty.device)
    sum_disc_price = torch.zeros(n, dtype=torch.float64, device=qty.device)
    sum_charge = torch.zeros(n, dtype=torch.float64, device=qty.device)
    avg_qty = torch.zeros(n, dtype=torch.float64, device=qty.device)
    avg_price = torch.zeros(n, dtype=torch.float64, device=qty.device)
    avg_disc = torch.zeros(n, dtype=torch.float64, device=qty.device)
    count = torch.zeros(n, dtype=torch.float64, device=qty.device)

    _K._q1_fused_kernel[grid](
        rf_codes, ls_codes, qty, price, disc, tax, ship_days,
        shipdate_hi_days, n,
        sum_qty, sum_base_price, sum_disc_price, sum_charge,
        avg_qty, avg_price, avg_disc, count,
    )

    # Group by and aggregate
    group_keys = rf_codes * 256 + ls_codes
    unique_keys, inverse = torch.unique(group_keys, return_inverse=True)
    n_groups = unique_keys.shape[0]

    # Initialize grouped outputs
    sum_qty_out = torch.zeros(n_groups, dtype=torch.float64, device=qty.device)
    sum_base_price_out = torch.zeros(n_groups, dtype=torch.float64, device=qty.device)
    sum_disc_price_out = torch.zeros(n_groups, dtype=torch.float64, device=qty.device)
    sum_charge_out = torch.zeros(n_groups, dtype=torch.float64, device=qty.device)
    count_out = torch.zeros(n_groups, dtype=torch.float64, device=qty.device)

    # Scatter-add
    sum_qty_out.scatter_add_(0, inverse, sum_qty)
    sum_base_price_out.scatter_add_(0, inverse, sum_base_price)
    sum_disc_price_out.scatter_add_(0, inverse, sum_disc_price)
    sum_charge_out.scatter_add_(0, inverse, sum_charge)
    count_out.scatter_add_(0, inverse, count)

    # Compute averages
    avg_qty_out = sum_qty_out / count_out
    avg_price_out = sum_base_price_out / count_out

    # For avg_disc, compute from raw discounts
    temp_avg_disc = torch.zeros(n_groups, dtype=torch.float64, device=qty.device)
    temp_avg_disc.scatter_add_(0, inverse, avg_disc)
    avg_disc_out = temp_avg_disc / count_out

    # Decode groups
    rf_out = unique_keys // 256
    ls_out = unique_keys % 256

    # Sort by group key
    group_keys_sorted = rf_out * 256 + ls_out
    sort_order = torch.argsort(group_keys_sorted)
    rf_out = rf_out[sort_order]
    ls_out = ls_out[sort_order]
    sum_qty_out = sum_qty_out[sort_order]
    sum_base_price_out = sum_base_price_out[sort_order]
    sum_disc_price_out = sum_disc_price_out[sort_order]
    sum_charge_out = sum_charge_out[sort_order]
    avg_qty_out = avg_qty_out[sort_order]
    avg_price_out = avg_price_out[sort_order]
    avg_disc_out = avg_disc_out[sort_order]
    count_out = count_out.to(torch.int64)[sort_order]

    rf_strs = [rf_categories[i] for i in rf_out.cpu().tolist()]
    ls_strs = [ls_categories[i] for i in ls_out.cpu().tolist()]

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
