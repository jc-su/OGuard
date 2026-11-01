import os
import tempfile
import importlib.util
import linecache
from typing import Any

import cudf
import pandas as pd
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q12_count_kernel(
    code_ptr,
    high_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    code = tl.load(code_ptr + offs, mask=mask, other=-1)
    high = tl.load(high_ptr + offs, mask=mask, other=0)

    is_high = high != 0
    is_code0 = mask & (code == 0)
    is_code1 = mask & (code == 1)

    one = tl.full([BLOCK_SIZE], 1, tl.int32)
    zero = tl.full([BLOCK_SIZE], 0, tl.int32)

    hi0 = tl.sum(tl.where(is_code0 & is_high, one, zero), axis=0).to(tl.int64)
    hi1 = tl.sum(tl.where(is_code1 & is_high, one, zero), axis=0).to(tl.int64)
    lo0 = tl.sum(tl.where(is_code0 & (~is_high), one, zero), axis=0).to(tl.int64)
    lo1 = tl.sum(tl.where(is_code1 & (~is_high), one, zero), axis=0).to(tl.int64)

    tl.atomic_add(out_ptr + 0, hi0)
    tl.atomic_add(out_ptr + 1, hi1)
    tl.atomic_add(out_ptr + 2, lo0)
    tl.atomic_add(out_ptr + 3, lo1)


@triton.jit
def _q12_filter_agg_kernel(
    code_ptr,
    high_ptr,
    commit_ptr,
    receipt_ptr,
    ship_ptr,
    receipt_lo_days,
    receipt_hi_days,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    code = tl.load(code_ptr + offs, mask=in_bounds, other=-1)
    high = tl.load(high_ptr + offs, mask=in_bounds, other=0)
    commit_d = tl.load(commit_ptr + offs, mask=in_bounds, other=0)
    receipt_d = tl.load(receipt_ptr + offs, mask=in_bounds, other=receipt_hi_days)
    ship_d = tl.load(ship_ptr + offs, mask=in_bounds, other=0)

    keep = (
        in_bounds
        & (commit_d < receipt_d)
        & (ship_d < commit_d)
        & (receipt_d >= receipt_lo_days)
        & (receipt_d < receipt_hi_days)
    )

    is_high = high != 0
    is_code0 = keep & (code == 0)
    is_code1 = keep & (code == 1)

    one = tl.full([BLOCK_SIZE], 1, tl.int32)
    zero = tl.full([BLOCK_SIZE], 0, tl.int32)

    hi0 = tl.sum(tl.where(is_code0 & is_high, one, zero), axis=0).to(tl.int64)
    hi1 = tl.sum(tl.where(is_code1 & is_high, one, zero), axis=0).to(tl.int64)
    lo0 = tl.sum(tl.where(is_code0 & (~is_high), one, zero), axis=0).to(tl.int64)
    lo1 = tl.sum(tl.where(is_code1 & (~is_high), one, zero), axis=0).to(tl.int64)

    tl.atomic_add(out_ptr + 0, hi0)
    tl.atomic_add(out_ptr + 1, hi1)
    tl.atomic_add(out_ptr + 2, lo0)
    tl.atomic_add(out_ptr + 3, lo1)
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q12_triton_")
    path = os.path.join(tmpdir, "_q12_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q12_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()
_BLOCK_SIZE: int = 4096


def _launch_q12_counts(
    l_shipmode_code: torch.Tensor,
    o_orderpriority_high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run on GPU with Triton: one fused fixed-group reduction over already-filtered joined rows.
    This avoids PyTorch boolean-mask materialization and scatter_add temporaries.
    """
    n = int(l_shipmode_code.numel())
    out = torch.zeros(4, dtype=torch.int64, device=l_shipmode_code.device)
    if n == 0:
        return out[0:2], out[2:4]

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q12_count_kernel[grid](
        l_shipmode_code,
        o_orderpriority_high,
        out,
        n,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return out[0:2], out[2:4]


def _launch_q12_filter_agg(
    l_shipmode_code: torch.Tensor,
    o_orderpriority_high: torch.Tensor,
    l_commitdate: torch.Tensor,
    l_receiptdate: torch.Tensor,
    l_shipdate: torch.Tensor,
    receipt_lo_days: int,
    receipt_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run on GPU with Triton: fused date predicates plus two-group count aggregation.
    This compatibility path replaces the eager Torch mask/scatter sequence with one scan kernel.
    """
    n = int(l_shipmode_code.numel())
    out = torch.zeros(4, dtype=torch.int64, device=l_shipmode_code.device)
    if n == 0:
        return out[0:2], out[2:4]

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q12_filter_agg_kernel[grid](
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        int(receipt_lo_days),
        int(receipt_hi_days),
        out,
        n,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return out[0:2], out[2:4]


def _query_core(
    l_shipmode_code: torch.Tensor,
    o_orderpriority_high: torch.Tensor,
    l_commitdate: torch.Tensor,
    l_receiptdate: torch.Tensor,
    l_shipdate: torch.Tensor,
    receipt_lo_days: int,
    receipt_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compatibility core for the reference tensor interface.
    Runs the hot filter/project/aggregate path in one Triton GPU kernel instead of eager Torch
    masks, gathers, casts, and scatter_add calls.
    """
    high_counts, low_counts = _launch_q12_filter_agg(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
    )
    group_ids = torch.arange(2, device=l_shipmode_code.device, dtype=torch.int64)
    return group_ids, high_counts, low_counts


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipmode1: str = "MAIL",
    shipmode2: str = "SHIP",
    receipt_lo: str = "1994-01-01",
    receipt_hi: str = "1995-01-01",
    priority1: str = "1-URGENT",
    priority2: str = "2-HIGH",
) -> cudf.DataFrame:
    """
    SQL:
      SELECT l_shipmode,
             SUM(CASE WHEN o_orderpriority = '1-URGENT'
                       OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
             SUM(CASE WHEN o_orderpriority <> '1-URGENT'
                       AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
      FROM orders, lineitem
      WHERE o_orderkey = l_orderkey
        AND l_shipmode IN ('MAIL', 'SHIP')
        AND l_commitdate < l_receiptdate
        AND l_shipdate < l_commitdate
        AND l_receiptdate >= DATE '1994-01-01'
        AND l_receiptdate < DATE '1995-01-01'
      GROUP BY l_shipmode
      ORDER BY l_shipmode;

    Description:
      TPC-H Q12. cuDF handles string predicates, int32 epoch-day prefiltering, and the hash
      join using only the required columns. The joined payload is reduced to two int32 columns
      (shipmode code and high-priority flag). A Triton GPU kernel then performs the fixed
      two-group aggregation in one coalesced scan, avoiding PyTorch mask/gather/scatter
      temporaries and reducing HBM traffic on the hot count path.

    Params:
      tables: Mapping containing cuDF DataFrames "orders" and "lineitem".
      shipmode1, shipmode2: Allowed shipping modes; output is ordered lexicographically.
      receipt_lo, receipt_hi: Inclusive/exclusive receipt-date bounds.
      priority1, priority2: Order-priority values counted as high priority.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    ship_mask = df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])
    li_ship = df_lineitem.loc[
        ship_mask,
        ["l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"],
    ]

    commit_days = ((li_ship["l_commitdate"].astype("int64") // 86400).astype("int32"))
    receipt_days = ((li_ship["l_receiptdate"].astype("int64") // 86400).astype("int32"))
    ship_days = ((li_ship["l_shipdate"].astype("int64") // 86400).astype("int32"))

    date_mask = (
        (commit_days < receipt_days)
        & (ship_days < commit_days)
        & (receipt_days >= receipt_lo_days)
        & (receipt_days < receipt_hi_days)
    )

    shipmode_code = (li_ship["l_shipmode"] == shipmode2).astype("int32")
    li_join = cudf.DataFrame(
        {
            "l_orderkey": li_ship["l_orderkey"][date_mask],
            "_shipmode_code": shipmode_code[date_mask],
        }
    )

    order_priority_high = (
        (df_orders["o_orderpriority"] == priority1)
        | (df_orders["o_orderpriority"] == priority2)
    ).astype("int32")
    orders_join = cudf.DataFrame(
        {
            "o_orderkey": df_orders["o_orderkey"],
            "_priority_high": order_priority_high,
        }
    )

    joined = li_join.merge(
        orders_join,
        left_on="l_orderkey",
        right_on="o_orderkey",
        how="inner",
    )

    shipmode_t = torch.as_tensor(joined["_shipmode_code"].to_cupy(), dtype=torch.int32)
    priority_t = torch.as_tensor(joined["_priority_high"].to_cupy(), dtype=torch.int32)

    high_counts, low_counts = _launch_q12_counts(shipmode_t, priority_t)

    names = [shipmode1, shipmode2]
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_counts.cpu().tolist()
    lc_host = low_counts.cpu().tolist()

    return cudf.DataFrame(
        {
            "l_shipmode": [names[i] for i in ordered],
            "high_line_count": [hc_host[i] for i in ordered],
            "low_line_count": [lc_host[i] for i in ordered],
        }
    )
