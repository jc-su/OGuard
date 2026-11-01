import os
import tempfile
import importlib.util
import linecache
from typing import Any

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q1_kernel(
    rf_ptr,
    ls_ptr,
    qty_ptr,
    price_ptr,
    disc_ptr,
    tax_ptr,
    shipdate_ptr,
    sum_qty_ptr,
    sum_base_price_ptr,
    sum_disc_price_ptr,
    sum_charge_ptr,
    sum_disc_ptr,
    count_ptr,
    n_elements,
    shipdate_hi_days,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi_days + 1)
    keep_date = mask & (shipdate <= shipdate_hi_days)

    rf = tl.load(rf_ptr + offs, mask=keep_date, other=-1)
    ls = tl.load(ls_ptr + offs, mask=keep_date, other=-1)

    qty = tl.load(qty_ptr + offs, mask=keep_date, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep_date, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep_date, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep_date, other=0.0)

    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    for r in tl.static_range(0, 4):
        for s in tl.static_range(0, 4):
            group_mask = keep_date & (rf == r) & (ls == s)
            idx = r * 4 + s

            tl.atomic_add(sum_qty_ptr + idx, tl.sum(tl.where(group_mask, qty, 0.0)))
            tl.atomic_add(sum_base_price_ptr + idx, tl.sum(tl.where(group_mask, price, 0.0)))
            tl.atomic_add(sum_disc_price_ptr + idx, tl.sum(tl.where(group_mask, disc_price, 0.0)))
            tl.atomic_add(sum_charge_ptr + idx, tl.sum(tl.where(group_mask, charge, 0.0)))
            tl.atomic_add(sum_disc_ptr + idx, tl.sum(tl.where(group_mask, disc, 0.0)))
            tl.atomic_add(count_ptr + idx, tl.sum(tl.where(group_mask, 1.0, 0.0)))
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q1_triton_")
    path = os.path.join(tmpdir, "_q1_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q1_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q1(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
    block_size: int = 4096,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """
    Runs the hot scan/filter/project/aggregate path on the GPU in one Triton kernel.

    The kernel reads the lineitem tensors once, applies the shipdate predicate in
    registers, computes discounted price and charge in registers, and performs
    per-program reductions into the small fixed TPC-H Q1 categorical group space.
    This avoids the reference eager PyTorch mask materialization, filtered column
    gathers, unique/inverse construction, and multiple scatter_add passes.
    """
    n = l_quantity.numel()
    device = l_quantity.device
    n_groups = 16

    sum_qty = torch.zeros(n_groups, dtype=torch.float64, device=device)
    sum_base_price = torch.zeros(n_groups, dtype=torch.float64, device=device)
    sum_disc_price = torch.zeros(n_groups, dtype=torch.float64, device=device)
    sum_charge = torch.zeros(n_groups, dtype=torch.float64, device=device)
    sum_disc = torch.zeros(n_groups, dtype=torch.float64, device=device)
    count_f = torch.zeros(n_groups, dtype=torch.float64, device=device)

    if n == 0:
        return sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc, count_f

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q1_kernel[grid](
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        sum_qty,
        sum_base_price,
        sum_disc_price,
        sum_charge,
        sum_disc,
        count_f,
        n,
        shipdate_hi_days,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc, count_f


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
    GPU tensor core for TPC-H Q1.

    cuDF work remains in run_query; this function operates only on already
    tensorized GPU columns. The memory-bound filter/project/aggregate scan is
    fused into a single Triton launch over the lineitem columns, with only tiny
    Torch tensor operations afterward to compact the 4x4 fixed categorical
    accumulator table and compute averages. The result layout and dtypes match
    the reference PyTorch implementation.
    """
    (
        all_sum_qty,
        all_sum_base_price,
        all_sum_disc_price,
        all_sum_charge,
        all_sum_disc,
        all_count_f,
    ) = _launch_q1(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        shipdate_hi_days,
    )

    nz = torch.nonzero(all_count_f > 0, as_tuple=False).flatten()

    sum_qty = all_sum_qty[nz]
    sum_base_price = all_sum_base_price[nz]
    sum_disc_price = all_sum_disc_price[nz]
    sum_charge = all_sum_charge[nz]
    sum_disc = all_sum_disc[nz]
    count_f = all_count_f[nz]
    count_order = count_f.to(torch.int64)

    avg_qty = sum_qty / count_f
    avg_price = sum_base_price / count_f
    avg_disc = sum_disc / count_f

    rf_out = (nz // 4).to(torch.int64)
    ls_out = (nz % 4).to(torch.int64)

    unique_keys = rf_out * 256 + ls_out
    order = torch.argsort(unique_keys)

    return (
        rf_out[order],
        ls_out[order],
        sum_qty[order],
        sum_base_price[order],
        sum_disc_price[order],
        sum_charge[order],
        avg_qty[order],
        avg_price[order],
        avg_disc[order],
        count_order[order],
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
