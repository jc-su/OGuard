import os
import tempfile
import importlib.util
import linecache
from types import ModuleType

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _max_i64_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.load(in_ptr + offs, mask=mask, other=0)
    m = tl.max(vals, axis=0)
    tl.store(out_ptr + pid, m)


@triton.jit
def _zero_u8_kernel(
    ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
    tl.store(ptr + offs, z, mask=mask)


@triton.jit
def _mark_late_orders_kernel(
    l_orderkey_ptr,
    l_commitdate_days_ptr,
    l_receiptdate_days_ptr,
    flags_ptr,
    n_elements,
    flag_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    commit_days = tl.load(l_commitdate_days_ptr + offs, mask=mask, other=0)
    receipt_days = tl.load(l_receiptdate_days_ptr + offs, mask=mask, other=0)
    late = mask & (commit_days < receipt_days)

    keys = tl.load(l_orderkey_ptr + offs, mask=late, other=-1)
    in_range = late & (keys >= 0) & (keys < flag_n)
    one = tl.full((BLOCK_SIZE,), 1, dtype=tl.uint8)
    tl.store(flags_ptr + keys, one, mask=in_range)


@triton.jit
def _count_q4_orders_kernel(
    o_orderkey_ptr,
    o_orderdate_days_ptr,
    o_orderpriority_codes_ptr,
    flags_ptr,
    counts_ptr,
    n_elements,
    date_lo_days,
    date_hi_days,
    flag_n,
    BLOCK_SIZE: tl.constexpr,
    NUM_PRIORITIES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    order_days = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=date_hi_days)
    date_keep = mask & (order_days >= date_lo_days) & (order_days < date_hi_days)

    keys = tl.load(o_orderkey_ptr + offs, mask=date_keep, other=-1)
    key_ok = date_keep & (keys >= 0) & (keys < flag_n)
    has_late_line = tl.load(flags_ptr + keys, mask=key_ok, other=0) != 0

    valid = key_ok & has_late_line
    prio = tl.load(o_orderpriority_codes_ptr + offs, mask=valid, other=-1)

    one_i64 = tl.full((BLOCK_SIZE,), 1, dtype=tl.int64)
    zero_i64 = tl.full((BLOCK_SIZE,), 0, dtype=tl.int64)

    for p in tl.static_range(0, NUM_PRIORITIES):
        c = tl.sum(tl.where(valid & (prio == p), one_i64, zero_i64), axis=0)
        tl.atomic_add(counts_ptr + p, c, sem="relaxed")
'''


def _load_kernels() -> ModuleType:
    """Write Triton kernels to a real Python file and import them.

    Runs on CPU at module import time. Triton JIT requires kernels to come from
    an on-disk source file under the validator, so all GPU kernels live in
    ``_KERNEL_SOURCE`` and are imported from a temporary module.
    """
    tmpdir = tempfile.mkdtemp(prefix="q4_triton_")
    path = os.path.join(tmpdir, "_q4_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q4_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _triton_max_i64(x: torch.Tensor, block_size: int = 1024) -> int:
    """Compute a non-negative int64 tensor maximum with Triton reductions.

    The reduction runs on GPU in one or more Triton launches, then transfers the
    final scalar to CPU only so the host can allocate the direct-address flag
    array. This replaces heavier PyTorch set-building work in the hot path.
    """
    n = x.numel()
    if n == 0:
        return 0

    cur = x
    cur_n = n
    while cur_n > 1:
        n_blocks = triton.cdiv(cur_n, block_size)
        partial = torch.empty(n_blocks, dtype=torch.int64, device=x.device)
        _K._max_i64_kernel[(n_blocks,)](
            cur,
            partial,
            cur_n,
            BLOCK_SIZE=block_size,
        )
        cur = partial
        cur_n = n_blocks

    return int(cur.item())


def _launch_q4_triton(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
    block_size_scan: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Q4 numeric hot path on GPU using Triton kernels.

    The lineitem scan marks a direct-address GPU byte flag for every orderkey
    with at least one late lineitem, then a fused orders scan applies the date
    predicate, performs the EXISTS probe from that flag array, and aggregates
    counts by small priority-code groups. This avoids Torch ``unique``/``isin``
    materialization and replaces multiple memory-bound passes with coalesced
    Triton scans plus a compact flag lookup.
    """
    device = o_orderkey.device

    max_l_key = _triton_max_i64(l_orderkey, block_size=block_size_scan) if l_orderkey.numel() else 0
    flag_n = max(max_l_key + 1, 1)

    late_order_flags = torch.empty(flag_n, dtype=torch.uint8, device=device)
    counts = torch.zeros(num_priorities, dtype=torch.int64, device=device)

    zero_grid = (triton.cdiv(flag_n, block_size_scan),)
    _K._zero_u8_kernel[zero_grid](
        late_order_flags,
        flag_n,
        BLOCK_SIZE=block_size_scan,
    )

    n_line = l_orderkey.numel()
    if n_line:
        line_grid = (triton.cdiv(n_line, block_size_scan),)
        _K._mark_late_orders_kernel[line_grid](
            l_orderkey,
            l_commitdate_days,
            l_receiptdate_days,
            late_order_flags,
            n_line,
            flag_n,
            BLOCK_SIZE=block_size_scan,
        )

    n_orders = o_orderkey.numel()
    if n_orders and num_priorities:
        order_grid = (triton.cdiv(n_orders, block_size_scan),)
        _K._count_q4_orders_kernel[order_grid](
            o_orderkey,
            o_orderdate_days,
            o_orderpriority_codes,
            late_order_flags,
            counts,
            n_orders,
            date_lo_days,
            date_hi_days,
            flag_n,
            BLOCK_SIZE=block_size_scan,
            NUM_PRIORITIES=num_priorities,
        )

    priority_ids = torch.arange(num_priorities, device=device, dtype=torch.int64)
    nz = counts > 0
    return priority_ids[nz], counts[nz]


def _query_core(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute TPC-H Q4 priority counts with Triton on GPU.

    This tensor-only core builds a GPU flag table of orderkeys having any
    qualifying lineitem and then fuses order-date filtering, EXISTS probing, and
    priority aggregation in Triton. The query is memory-bound, so direct flag
    probes and fused scans avoid the expensive Torch ``unique``/``isin`` path
    and reduce temporary HBM traffic.
    """
    return _launch_q4_triton(
        o_orderkey,
        o_orderdate_days,
        o_orderpriority_codes,
        l_orderkey,
        l_commitdate_days,
        l_receiptdate_days,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count FROM orders WHERE o_orderdate >= CAST('1993-07-01' AS date) AND o_orderdate < CAST('1993-10-01' AS date) AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority;
    Description: TPC-H Q4 - Counts orders per priority within a date range where at least one lineitem shipped late (commit before receipt).
    Params:
        date_lo (str): inclusive lower bound for o_orderdate.
        date_hi (str): exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    # Encode o_orderpriority as categorical for groupby on GPU
    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes  # int codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)
    o_orderdate_days_t = torch.as_tensor(
        ((df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    o_priority_codes_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32)

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_commitdate_days_t = torch.as_tensor(
        ((df_lineitem["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_receiptdate_days_t = torch.as_tensor(
        ((df_lineitem["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    priority_ids_t, counts_t = _query_core(
        o_orderkey_t,
        o_orderdate_days_t,
        o_priority_codes_t,
        l_orderkey_t,
        l_commitdate_days_t,
        l_receiptdate_days_t,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )

    # Map codes back to priority strings and sort
    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[i] for i in priority_ids_host]

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
        "order_count": counts_t,
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
