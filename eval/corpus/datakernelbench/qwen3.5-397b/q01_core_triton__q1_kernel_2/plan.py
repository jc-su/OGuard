import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton
import triton.language as tl

# Triton kernel source for fused filter + group-by aggregation
# Single pass: filter by shipdate, compute key and aggregates, atomic_add to fixed buffer
# Key space: rf (0-255) * 256 + ls (0-255) = 0-65535
# Accumulator layout per key: [sum_qty, sum_price, sum_disc_price, sum_charge, sum_disc, count]
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
    reset_to_zero=["acc_ptr"],
)
@triton.jit
def _q1_kernel(
    rf_ptr, ls_ptr,
    qty_ptr, price_ptr, disc_ptr, tax_ptr, shipdate_ptr,
    shipdate_hi,
    acc_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load shipdate first for early filter
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi + 1)
    keep = mask & (shipdate <= shipdate_hi)

    # Load group keys (only meaningful if keep)
    rf = tl.load(rf_ptr + offs, mask=mask, other=0).to(tl.int64)
    ls = tl.load(ls_ptr + offs, mask=mask, other=0).to(tl.int64)
    key = rf * 256 + ls

    # Load measure columns (conditional on keep)
    qty = tl.load(qty_ptr + offs, mask=keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    # Compute derived aggregates in registers
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    # Accumulate to global buffer using atomics
    # acc_ptr layout: [key * 6 + col_idx]
    # cols: 0:qty, 1:price, 2:disc_price, 3:charge, 4:disc, 5:count
    base = key * 6

    # Use tl.where to zero out non-kept rows for atomic_add
    tl.atomic_add(acc_ptr + base + 0, tl.where(keep, qty, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 1, tl.where(keep, price, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 2, tl.where(keep, disc_price, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 3, tl.where(keep, charge, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 4, tl.where(keep, disc, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 5, tl.where(keep, 1.0, 0.0), sem="acq_rel")
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

def _query_core(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """
    Fused Triton kernel for TPC-H Q1 aggregation.
    Single GPU pass: filter + project + aggregate via atomics to fixed buffer.
    Avoids multiple scatter_add calls and temporary filtered tensors.
    """
    n = l_quantity.numel()
    device = l_quantity.device

    # Fixed accumulator buffer: 65536 keys * 6 columns (float64)
    # 65536 * 6 * 8 = 3MB, fits easily in HBM
    acc = torch.zeros(65536 * 6, dtype=torch.float64, device=device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q1_kernel[grid](
        l_returnflag_code, l_linestatus_code,
        l_quantity, l_extendedprice, l_discount, l_tax, l_shipdate_days,
        shipdate_hi_days,
        acc,
        n,
    )

    # Reshape to (65536, 6)
    acc = acc.view(65536, 6)

    # Find valid groups (count > 0)
    counts = acc[:, 5]
    valid_mask = counts > 0.0
    valid_keys = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

    if valid_keys.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        )

    # Sort by key (already sorted from nonzero, but ensure)
    order = torch.argsort(valid_keys)
    sorted_keys = valid_keys[order]

    # Decode keys to rf, ls
    rf_out = (sorted_keys // 256).to(torch.int64)
    ls_out = (sorted_keys % 256).to(torch.int64)

    # Extract aggregates
    sum_qty = acc[sorted_keys, 0]
    sum_base_price = acc[sorted_keys, 1]
    sum_disc_price = acc[sorted_keys, 2]
    sum_charge = acc[sorted_keys, 3]
    sum_disc = acc[sorted_keys, 4]
    count_order_f = acc[sorted_keys, 5]
    count_order = count_order_f.to(torch.int64)

    # Compute averages
    avg_qty = sum_qty / count_order_f
    avg_price = sum_base_price / count_order_f
    avg_disc = sum_disc / count_order_f

    return (
        rf_out,
        ls_out,
        sum_qty,
        sum_base_price,
        sum_disc_price,
        sum_charge,
        avg_qty,
        avg_price,
        avg_disc,
        count_order,
    )


def run_query(tables: dict[str, cudf.DataFrame], shipdate_hi: str = "1998-09-02") -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc,
         count(*) AS count_order FROM lineitem WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;
    Description: TPC-H Q1 pricing summary report aggregated by return flag and line status.
    Params: shipdate_hi (str): upper-bound ship date filter.
    """
    df_lineitem = tables["lineitem"]

    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Encode string columns l_returnflag, l_linestatus to integer codes (single-char strings).
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    # Use categorical codes for encoding
    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    l_returnflag_code_t = torch.as_tensor(rf_codes.cat.codes.astype("int32").to_cupy())
    l_linestatus_code_t = torch.as_tensor(ls_codes.cat.codes.astype("int32").to_cupy())

    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_tax_t = torch.as_tensor(df_lineitem["l_tax"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    (
        rf_out, ls_out, sum_qty, sum_base_price, sum_disc_price,
        sum_charge, avg_qty, avg_price, avg_disc, count_order,
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
    )

    # Decode category codes back to strings
    rf_out_host = rf_out.cpu().tolist()
    ls_out_host = ls_out.cpu().tolist()
    rf_strs = [rf_categories[i] for i in rf_out_host]
    ls_strs = [ls_categories[i] for i in ls_out_host]

    return cudf.DataFrame({
        "l_returnflag": cudf.Series(rf_strs),
        "l_linestatus": cudf.Series(ls_strs),
        "sum_qty": sum_qty,
        "sum_base_price": sum_base_price,
        "sum_disc_price": sum_disc_price,
        "sum_charge": sum_charge,
        "avg_qty": avg_qty,
        "avg_price": avg_price,
        "avg_disc": avg_disc,
        "count_order": count_order,
    })
