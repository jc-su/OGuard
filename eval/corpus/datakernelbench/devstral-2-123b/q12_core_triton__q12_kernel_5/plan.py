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
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["high_out_ptr", "low_out_ptr"],
)
@triton.jit
def _q12_kernel(
    l_commitdate_ptr, l_receiptdate_ptr, l_shipdate_ptr, shipmode_ptr,
    o_orderpriority_ptr, receipt_lo, receipt_hi, n_elements,
    high_out_ptr, low_out_ptr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load date columns
    commit = tl.load(l_commitdate_ptr + offs, mask=mask, other=receipt_hi)
    receipt = tl.load(l_receiptdate_ptr + offs, mask=mask, other=receipt_hi)
    ship = tl.load(l_shipdate_ptr + offs, mask=mask, other=receipt_hi)

    # Filter condition
    keep = (
        mask
        & (commit < receipt)
        & (ship < commit)
        & (receipt >= receipt_lo)
        & (receipt < receipt_hi)
    )

    # Load shipmode and priority for kept rows
    shipmode = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    priority = tl.load(o_orderpriority_ptr + offs, mask=keep, other=0)

    # Compute group counts using block-level accumulation
    for sm in tl.static_range(2):
        sm_mask = (shipmode == sm) & keep
        high_mask = sm_mask & (priority > 0)

        # Reduce within warp
        high_sum = tl.sum(high_mask, axis=0)
        low_sum = tl.sum(sm_mask, axis=0) - high_sum

        # Store results using atomic add
        tl.atomic_add(high_out_ptr + sm, high_sum)
        tl.atomic_add(low_out_ptr + sm, low_sum)
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

def _run_q12_kernel(
    l_commitdate, l_receiptdate, l_shipdate, shipmode, priority,
    receipt_lo, receipt_hi
):
    n = l_commitdate.numel()
    high_counts = torch.zeros(2, dtype=torch.int64, device=l_commitdate.device)
    low_counts = torch.zeros(2, dtype=torch.int64, device=l_commitdate.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q12_kernel[grid](
        l_commitdate, l_receiptdate, l_shipdate, shipmode, priority,
        receipt_lo, receipt_hi, n,
        high_counts, low_counts
    )

    return high_counts, low_counts

def _query_core(
    l_commitdate: torch.Tensor,
    l_receiptdate: torch.Tensor,
    l_shipdate: torch.Tensor,
    shipmode: torch.Tensor,
    priority: torch.Tensor,
    receipt_lo_days: int,
    receipt_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_ids = torch.tensor([0, 1], device=shipmode.device, dtype=torch.int64)
    high_counts, low_counts = _run_q12_kernel(
        l_commitdate, l_receiptdate, l_shipdate, shipmode, priority,
        receipt_lo_days, receipt_hi_days
    )
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
        l_commit_t,
        l_receipt_t,
        l_ship_t,
        shipmode_t,
        priority_t,
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
