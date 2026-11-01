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
    reset_to_zero=["high_counts_ptr", "low_counts_ptr"],
)
@triton.jit
def _q12_kernel(
    shipmode_ptr, priority_ptr, commit_ptr, receipt_ptr, ship_ptr,
    receipt_lo, receipt_hi,
    high_counts_ptr, low_counts_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Evaluate receiptdate first (selectivity ~15% for the year filter)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=0)
    keep = mask & (receipt >= receipt_lo) & (receipt < receipt_hi)

    # Conditionally load commitdate
    commit = tl.load(commit_ptr + offs, mask=keep, other=0)
    keep = keep & (commit < receipt)

    # Conditionally load shipdate
    ship = tl.load(ship_ptr + offs, mask=keep, other=0)
    keep = keep & (ship < commit)

    # Conditionally load shipmode and priority (encoded as int32)
    shipmode = tl.load(shipmode_ptr + offs, mask=keep, other=-1)
    priority = tl.load(priority_ptr + offs, mask=keep, other=-1)

    is_mail = (shipmode == 0)
    is_ship = (shipmode == 1)
    is_high = (priority == 1)
    is_low  = (priority == 0)

    # Compute block-local sums for each category and group
    c_h0 = tl.sum(tl.where(keep & is_mail & is_high, 1, 0)).to(tl.int64)
    c_l0 = tl.sum(tl.where(keep & is_mail & is_low,  1, 0)).to(tl.int64)
    c_h1 = tl.sum(tl.where(keep & is_ship & is_high, 1, 0)).to(tl.int64)
    c_l1 = tl.sum(tl.where(keep & is_ship & is_low,  1, 0)).to(tl.int64)

    # Global accumulation
    tl.atomic_add(high_counts_ptr + 0, c_h0)
    tl.atomic_add(low_counts_ptr + 0, c_l0)
    tl.atomic_add(high_counts_ptr + 1, c_h1)
    tl.atomic_add(low_counts_ptr + 1, c_l1)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q12_triton_")
    path = os.path.join(tmpdir, "_q12_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q12_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

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
    Executes a single-pass fused Triton kernel for memory-bound filtering and aggregation.
    Uses conditional loads heavily to avoid HBM read traffic when rows fail early predicates.
    Reduces memory usage by eliminating eager boolean intermediate masks entirely.
    """
    n = l_shipmode_code.numel()
    
    # Aggregation buffers - initialize to zero (0 = MAIL, 1 = SHIP)
    high_counts = torch.zeros(2, dtype=torch.int64, device=l_shipmode_code.device)
    low_counts = torch.zeros(2, dtype=torch.int64, device=l_shipmode_code.device)

    if n > 0:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _K._q12_kernel[grid](
            l_shipmode_code,
            o_orderpriority_high,
            l_commitdate,
            l_receiptdate,
            l_shipdate,
            receipt_lo_days,
            receipt_hi_days,
            high_counts,
            low_counts,
            n
        )

    # Two shipmode groups: 0 = MAIL, 1 = SHIP
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
