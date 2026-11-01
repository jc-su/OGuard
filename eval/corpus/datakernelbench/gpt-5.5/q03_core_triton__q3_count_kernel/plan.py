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
def _q3_count_kernel(
    l_shipdate_days_ptr,
    o_orderdate_days_ptr,
    keep_ptr,
    counts_ptr,
    shipdate_lo_days,
    orderdate_hi_days,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    l_shipdate = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_lo_days)
    o_orderdate = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=orderdate_hi_days)

    keep = mask & (l_shipdate > shipdate_lo_days) & (o_orderdate < orderdate_hi_days)
    tl.store(keep_ptr + offs, keep.to(tl.uint8), mask=mask)

    cnt = tl.sum(keep.to(tl.int64), axis=0)
    tl.store(counts_ptr + pid, cnt)


@triton.jit
def _q3_scatter_kernel(
    l_orderkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    o_orderdate_days_ptr,
    o_shippriority_ptr,
    keep_ptr,
    block_offsets_ptr,
    out_lk_ptr,
    out_rev_ptr,
    out_od_ptr,
    out_sp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    keep_u8 = tl.load(keep_ptr + offs, mask=in_bounds, other=0)
    keep = keep_u8 != 0

    local_rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    base = tl.load(block_offsets_ptr + pid)
    out_idx = base + local_rank

    lk = tl.load(l_orderkey_ptr + offs, mask=keep, other=0)
    od = tl.load(o_orderdate_days_ptr + offs, mask=keep, other=0)
    sp = tl.load(o_shippriority_ptr + offs, mask=keep, other=0)
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    rev = price * (1.0 - disc)

    tl.store(out_lk_ptr + out_idx, lk, mask=keep)
    tl.store(out_rev_ptr + out_idx, rev, mask=keep)
    tl.store(out_od_ptr + out_idx, od, mask=keep)
    tl.store(out_sp_ptr + out_idx, sp, mask=keep)
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q3_triton_")
    path = os.path.join(tmpdir, "_q3_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q3_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _compact_q3_rows(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU/Triton phase for Q3 row filtering and projection.

    The first Triton kernel evaluates the two date predicates in one coalesced pass
    and counts selected rows per block.  The second Triton kernel compacts selected
    rows and computes revenue in registers, loading extended price/discount only
    for rows that survive the date filter.  This avoids the reference's eager Torch
    boolean-index temporaries for all projected columns and reduces HBM traffic on
    the memory-bound filter/project part of the query core.
    """
    n = int(l_orderkey.numel())
    device = l_orderkey.device
    if n == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.float64, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
        )

    num_blocks = triton.cdiv(n, block_size)
    keep = torch.empty((n,), dtype=torch.uint8, device=device)
    counts = torch.empty((num_blocks,), dtype=torch.int64, device=device)

    _K._q3_count_kernel[(num_blocks,)](
        l_shipdate_days,
        o_orderdate_days,
        keep,
        counts,
        int(shipdate_lo_days),
        int(orderdate_hi_days),
        n,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    total = int(counts.sum().item())
    out_lk = torch.empty((total,), dtype=torch.int64, device=device)
    out_rev = torch.empty((total,), dtype=torch.float64, device=device)
    out_od = torch.empty((total,), dtype=torch.int32, device=device)
    out_sp = torch.empty((total,), dtype=torch.int32, device=device)

    if total == 0:
        return out_lk, out_rev, out_od, out_sp

    block_offsets = torch.cumsum(counts, dim=0) - counts

    _K._q3_scatter_kernel[(num_blocks,)](
        l_orderkey,
        l_extendedprice,
        l_discount,
        o_orderdate_days,
        o_shippriority,
        keep,
        block_offsets,
        out_lk,
        out_rev,
        out_od,
        out_sp,
        n,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    return out_lk, out_rev, out_od, out_sp


def _aggregate_topn_q3(
    lk: torch.Tensor,
    rev: torch.Tensor,
    od: torch.Tensor,
    sp: torch.Tensor,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU tensor phase for GROUP BY and ORDER BY after Triton compaction.

    The variable-cardinality GROUP BY is handled with GPU Torch sort/segment
    primitives because it is not a small fixed reduction.  Rows are lexicographically
    ordered by (l_orderkey, o_orderdate, o_shippriority), reduced by segment, then
    stably ordered by revenue descending and order date ascending to match the
    reference result ordering.
    """
    m = int(lk.numel())
    device = lk.device
    if m == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.float64, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
        )

    idx_sp = torch.argsort(sp, stable=True)
    lk_s = lk[idx_sp]
    rev_s = rev[idx_sp]
    od_s = od[idx_sp]
    sp_s = sp[idx_sp]

    idx_od = torch.argsort(od_s, stable=True)
    lk_s = lk_s[idx_od]
    rev_s = rev_s[idx_od]
    od_s = od_s[idx_od]
    sp_s = sp_s[idx_od]

    idx_lk = torch.argsort(lk_s, stable=True)
    lk_s = lk_s[idx_lk]
    rev_s = rev_s[idx_lk]
    od_s = od_s[idx_lk]
    sp_s = sp_s[idx_lk]

    is_start = torch.empty((m,), dtype=torch.bool, device=device)
    is_start[0] = True
    if m > 1:
        is_start[1:] = (
            (lk_s[1:] != lk_s[:-1])
            | (od_s[1:] != od_s[:-1])
            | (sp_s[1:] != sp_s[:-1])
        )

    seg_ids = torch.cumsum(is_start.to(torch.int64), dim=0) - 1
    num_groups = int(seg_ids[-1].item()) + 1

    revenue_sum = torch.zeros((num_groups,), dtype=torch.float64, device=device)
    revenue_sum.scatter_add_(0, seg_ids, rev_s)

    starts = torch.nonzero(is_start, as_tuple=False).flatten()
    u_lk = lk_s[starts]
    u_od = od_s[starts].to(torch.int32)
    u_sp = sp_s[starts].to(torch.int32)

    k = min(int(top_n), num_groups)

    idx_sec = torch.argsort(u_od, stable=True)
    rev_by_date = revenue_sum[idx_sec]
    lk_by_date = u_lk[idx_sec]
    od_by_date = u_od[idx_sec]
    sp_by_date = u_sp[idx_sec]

    idx_pri = torch.argsort(rev_by_date, descending=True, stable=True)

    rev_sorted = rev_by_date[idx_pri][:k]
    lk_sorted = lk_by_date[idx_pri][:k]
    od_sorted = od_by_date[idx_pri][:k]
    sp_sorted = sp_by_date[idx_pri][:k]

    return lk_sorted, rev_sorted, od_sorted, sp_sorted


def _query_core(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the numeric GPU core of TPC-H Q3.

    Triton kernels run the memory-bound date filter, revenue projection, and row
    compaction in two GPU passes with coalesced loads and fewer large Torch boolean
    indexing temporaries.  The remaining variable-cardinality aggregation and top-N
    ordering run as GPU tensor operations over the compacted rows.
    """
    lk, rev, od, sp = _compact_q3_rows(
        l_orderkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        o_orderdate_days,
        o_shippriority,
        shipdate_lo_days,
        orderdate_hi_days,
    )
    return _aggregate_topn_q3(lk, rev, od, sp, int(top_n))


def run_query(
    tables: dict[str, cudf.DataFrame],
    mktsegment: str = "BUILDING",
    orderdate_hi: str = "1995-03-15",
    shipdate_lo: str = "1995-03-15",
    top_n: int = 10,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate,
               o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < CAST('1995-03-15' AS date)
          AND l_shipdate > CAST('1995-03-15' AS date)
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;
    Description: TPC-H Q3 - Shipping Priority. Joins customer, orders, lineitem;
                 filters by market segment and date predicates; aggregates revenue
                 per (orderkey, orderdate, shippriority); returns top-N by revenue.
    Params:
        mktsegment (str): customer market segment filter.
        orderdate_hi (str): exclusive upper bound on o_orderdate.
        shipdate_lo (str): exclusive lower bound on l_shipdate.
        top_n (int): LIMIT value.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Literal normalization: string dates -> epoch-day ints
    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    # Filter customers by mktsegment
    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]

    # Join customer -> orders
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")

    # Join (customer, orders) -> lineitem
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Convert date columns to int32 day-counts
    o_orderdate_days = (
        (col["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    l_shipdate_days = (
        (col["l_shipdate"].astype("int64") // 86400).astype("int32")
    )

    lk_t, rev_t, od_t, sp_t = _query_core(
        torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32),
        shipdate_lo_days,
        orderdate_hi_days,
        int(top_n),
    )

    # Convert orderdate back to datetime64[s] for output
    od_series = cudf.Series(od_t.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_t,
        "revenue": rev_t,
        "o_orderdate": od_series,
        "o_shippriority": sp_t,
    })
