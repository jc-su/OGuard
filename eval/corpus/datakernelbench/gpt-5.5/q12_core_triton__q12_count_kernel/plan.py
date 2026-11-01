import os
import tempfile
import importlib.util
import linecache
import types

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q12_count_kernel(
    shipmode_code_ptr,
    priority_high_ptr,
    commitdate_ptr,
    receiptdate_ptr,
    shipdate_ptr,
    receipt_lo_days,
    receipt_hi_days,
    high_out_ptr,
    low_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offs < n_elements

    receipt = tl.load(receiptdate_ptr + offs, mask=active, other=receipt_lo_days - 1)
    commit = tl.load(commitdate_ptr + offs, mask=active, other=0)
    ship = tl.load(shipdate_ptr + offs, mask=active, other=0)

    keep = (
        active
        & (commit < receipt)
        & (ship < commit)
        & (receipt >= receipt_lo_days)
        & (receipt < receipt_hi_days)
    )

    code = tl.load(shipmode_code_ptr + offs, mask=keep, other=-1)
    high = tl.load(priority_high_ptr + offs, mask=keep, other=0)

    is_mail = keep & (code == 0)
    is_ship = keep & (code == 1)

    high_mail = tl.sum(tl.where(is_mail, high, 0), axis=0)
    high_ship = tl.sum(tl.where(is_ship, high, 0), axis=0)
    low_mail = tl.sum(tl.where(is_mail, 1 - high, 0), axis=0)
    low_ship = tl.sum(tl.where(is_ship, 1 - high, 0), axis=0)

    tl.atomic_add(high_out_ptr + 0, high_mail.to(tl.int64), sem="relaxed")
    tl.atomic_add(high_out_ptr + 1, high_ship.to(tl.int64), sem="relaxed")
    tl.atomic_add(low_out_ptr + 0, low_mail.to(tl.int64), sem="relaxed")
    tl.atomic_add(low_out_ptr + 1, low_ship.to(tl.int64), sem="relaxed")
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton GPU kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q12_triton_")
    path = os.path.join(tmpdir, "_q12_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q12_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q12_count_kernel(
    l_shipmode_code: torch.Tensor,
    o_orderpriority_high: torch.Tensor,
    l_commitdate: torch.Tensor,
    l_receiptdate: torch.Tensor,
    l_shipdate: torch.Tensor,
    receipt_lo_days: int,
    receipt_hi_days: int,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one fused Triton GPU pass over joined tensors.

    The kernel evaluates all date predicates, projects the two fixed shipmode groups,
    and accumulates high/low priority counts into two int64 output vectors. This is
    faster than the eager Torch reference because it avoids materializing a boolean
    mask, filtered columns, and scatter-add temporaries, reducing HBM traffic for this
    memory-bound scan.
    """
    n = int(l_shipmode_code.numel())
    device = l_shipmode_code.device

    group_ids = torch.arange(2, device=device, dtype=torch.int64)
    high_counts = torch.zeros(2, dtype=torch.int64, device=device)
    low_counts = torch.zeros(2, dtype=torch.int64, device=device)

    if n == 0:
        return group_ids, high_counts, low_counts

    if not l_shipmode_code.is_contiguous():
        l_shipmode_code = l_shipmode_code.contiguous()
    if not o_orderpriority_high.is_contiguous():
        o_orderpriority_high = o_orderpriority_high.contiguous()
    if not l_commitdate.is_contiguous():
        l_commitdate = l_commitdate.contiguous()
    if not l_receiptdate.is_contiguous():
        l_receiptdate = l_receiptdate.contiguous()
    if not l_shipdate.is_contiguous():
        l_shipdate = l_shipdate.contiguous()

    grid = (triton.cdiv(n, block_size),)
    _K._q12_count_kernel[grid](
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
        high_counts,
        low_counts,
        n,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return group_ids, high_counts, low_counts


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
    GPU tensor core for TPC-H Q12 aggregation.

    All numeric work runs in a fused Triton GPU kernel: date filtering, two-group
    projection, and int64 high/low count aggregation. cuDF join/string preparation
    remains in run_query; this function replaces the eager PyTorch mask/filter/scatter
    hot path with one coalesced memory-bound scan and four per-block atomic adds.
    """
    return _launch_q12_count_kernel(
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
    )


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
    SQL: SELECT l_shipmode, SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
         SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
         FROM orders, lineitem WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL','SHIP')
         AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
         AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01'
         GROUP BY l_shipmode ORDER BY l_shipmode;
    Description: TPC-H Q12 - counts urgent vs non-urgent orders by ship mode.
    Params:
      shipmode1, shipmode2 (str): allowed shipping modes.
      receipt_lo, receipt_hi (str): receipt date range.
      priority1, priority2 (str): high-priority order priority values.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    # Pre-filter lineitem on shipmode (reduces join size)
    li = df_lineitem[df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])]

    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey")

    # Encode shipmode: MAIL=0, SHIP=1
    shipmode_code = (joined["l_shipmode"] == shipmode2).astype("int32")
    priority_high = (
        (joined["o_orderpriority"] == priority1)
        | (joined["o_orderpriority"] == priority2)
    ).astype("int32")

    l_commit_t = torch.as_tensor(
        ((joined["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_receipt_t = torch.as_tensor(
        ((joined["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_ship_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    shipmode_t = torch.as_tensor(shipmode_code.to_cupy(), dtype=torch.int32)
    priority_t = torch.as_tensor(priority_high.to_cupy(), dtype=torch.int32)

    group_ids, high_counts, low_counts = _query_core(
        shipmode_t,
        priority_t,
        l_commit_t,
        l_receipt_t,
        l_ship_t,
        receipt_lo_days,
        receipt_hi_days,
    )

    # Decode group ids back to strings; order alphabetically (MAIL < SHIP)
    names = [shipmode1, shipmode2]  # index 0 -> MAIL, 1 -> SHIP
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_counts.cpu().tolist()
    lc_host = low_counts.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count": [lc_host[i] for i in ordered],
    })
    return result
