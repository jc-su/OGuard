from __future__ import annotations

import importlib.util
import linecache
import os
import tempfile
import types

import cudf
import pandas as pd
import torch
import triton

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _q1_group_kernel(
    rf_ptr,
    ls_ptr,
    qty_ptr,
    price_ptr,
    disc_ptr,
    tax_ptr,
    shipdate_ptr,
    date_hi,
    out_ptr,
    n_elements,
    N_RF: tl.constexpr,
    N_LS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=valid, other=date_hi + 1)
    keep = valid & (shipdate <= date_hi)

    rf = tl.load(rf_ptr + offs, mask=keep, other=-1)
    ls = tl.load(ls_ptr + offs, mask=keep, other=-1)

    qty = tl.load(qty_ptr + offs, mask=keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)
    one64 = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    zero64 = tl.full((BLOCK_SIZE,), 0.0, tl.float64)

    # out layout per group: [sum_qty, sum_base_price, sum_disc_price,
    #                        sum_charge, sum_disc, count_as_fp64, reserved]
    for r in tl.static_range(0, N_RF):
        for s in tl.static_range(0, N_LS):
            gmask = keep & (rf == r) & (ls == s)
            base = (r * N_LS + s) * 7

            sum_qty = tl.sum(tl.where(gmask, qty, zero64))
            sum_base = tl.sum(tl.where(gmask, price, zero64))
            sum_disc_price = tl.sum(tl.where(gmask, disc_price, zero64))
            sum_charge = tl.sum(tl.where(gmask, charge, zero64))
            sum_disc = tl.sum(tl.where(gmask, disc, zero64))
            cnt = tl.sum(tl.where(gmask, one64, zero64))

            tl.atomic_add(out_ptr + base + 0, sum_qty, sem="relaxed")
            tl.atomic_add(out_ptr + base + 1, sum_base, sem="relaxed")
            tl.atomic_add(out_ptr + base + 2, sum_disc_price, sem="relaxed")
            tl.atomic_add(out_ptr + base + 3, sum_charge, sem="relaxed")
            tl.atomic_add(out_ptr + base + 4, sum_disc, sem="relaxed")
            tl.atomic_add(out_ptr + base + 5, cnt, sem="relaxed")
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton JIT source to a real file and import it for validator-safe compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q1_triton_")
    path = os.path.join(tmpdir, "_q1_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)

    spec = importlib.util.spec_from_file_location("_q1_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q1_group_kernel(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
    n_returnflag_categories: int,
    n_linestatus_categories: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Run the bulk scan/aggregate on GPU in one fused Triton kernel.

    The kernel reads ship date, category codes, and numeric columns once, applies the
    date predicate in registers, computes Q1 projections, and atomically accumulates
    per small fixed group. This avoids the reference path's filtered temporaries,
    torch.unique, inverse indices, and multiple scatter_add passes, which is faster
    for this memory-bound TPC-H lineitem scan.
    """
    n_elements = int(l_quantity.numel())
    n_groups = max(int(n_returnflag_categories) * int(n_linestatus_categories), 1)
    out = torch.zeros(n_groups * 7, dtype=torch.float64, device=l_quantity.device)

    if n_elements == 0 or n_returnflag_categories <= 0 or n_linestatus_categories <= 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _K._q1_group_kernel[grid](
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        int(shipdate_hi_days),
        out,
        n_elements,
        N_RF=int(n_returnflag_categories),
        N_LS=int(n_linestatus_categories),
        BLOCK_SIZE=int(block_size),
        num_warps=8,
    )
    return out


def _query_core(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
    n_returnflag_categories: int,
    n_linestatus_categories: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    GPU implementation of the Q1 core.

    Triton performs the hot filter/project/group aggregate in one fused GPU pass.
    PyTorch is used only for tiny post-processing over the small group table:
    selecting non-empty groups, computing averages, and producing sorted group codes.
    """
    flat = _launch_q1_group_kernel(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        shipdate_hi_days,
        n_returnflag_categories,
        n_linestatus_categories,
    )

    n_groups = max(int(n_returnflag_categories) * int(n_linestatus_categories), 1)
    agg = flat.view(n_groups, 7)
    count_f_all = agg[:, 5]
    group_idx = torch.nonzero(count_f_all > 0.0, as_tuple=False).flatten()

    sum_qty = agg[:, 0][group_idx]
    sum_base_price = agg[:, 1][group_idx]
    sum_disc_price = agg[:, 2][group_idx]
    sum_charge = agg[:, 3][group_idx]
    sum_disc = agg[:, 4][group_idx]
    count_f = count_f_all[group_idx]

    avg_qty = sum_qty / count_f
    avg_price = sum_base_price / count_f
    avg_disc = sum_disc / count_f
    count_order = count_f.to(torch.int64)

    rf_out = torch.div(group_idx, int(n_linestatus_categories), rounding_mode="floor").to(torch.int64)
    ls_out = (group_idx % int(n_linestatus_categories)).to(torch.int64)

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
    SQL:
        SELECT
            l_returnflag,
            l_linestatus,
            sum(l_quantity) AS sum_qty,
            sum(l_extendedprice) AS sum_base_price,
            sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            avg(l_quantity) AS avg_qty,
            avg(l_extendedprice) AS avg_price,
            avg(l_discount) AS avg_disc,
            count(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= CAST('1998-09-02' AS date)
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus;

    Description:
        TPC-H Q1 pricing summary report. cuDF handles string categorical encoding,
        date conversion to int32 epoch days, and final cudf.DataFrame construction.
        The memory-bound hot path over lineitem is a fused Triton GPU scan that
        applies the ship-date filter, computes derived prices, and aggregates all
        fixed small groups in one pass, reducing temporary tensors and repeated HBM
        traffic versus eager PyTorch filtering, unique, and scatter_add operations.

    Params:
        tables: Mapping containing the cuDF table "lineitem".
        shipdate_hi: Inclusive upper-bound ship date as a string parseable by pandas.
    """
    df_lineitem = tables["lineitem"]
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

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
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
        len(rf_categories),
        len(ls_categories),
    )

    rf_out_host = rf_out.cpu().tolist()
    ls_out_host = ls_out.cpu().tolist()
    rf_strs = [rf_categories[int(i)] for i in rf_out_host]
    ls_strs = [ls_categories[int(i)] for i in ls_out_host]

    return cudf.DataFrame(
        {
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
        }
    )
