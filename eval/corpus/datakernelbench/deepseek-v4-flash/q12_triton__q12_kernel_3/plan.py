import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
import numpy as np

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
    reset_to_zero=["out_high_ptr", "out_low_ptr"],
)
@triton.jit
def _q12_kernel(
    shipmode_ptr, priority_ptr, commitdate_ptr, receiptdate_ptr, shipdate_ptr,
    receipt_lo, receipt_hi, out_high_ptr, out_low_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load filter columns
    commit = tl.load(commitdate_ptr + offs, mask=mask, other=receipt_hi)
    receipt = tl.load(receiptdate_ptr + offs, mask=mask, other=receipt_lo)
    ship = tl.load(shipdate_ptr + offs, mask=mask, other=receipt_hi)
    
    # Predicates
    keep = mask
    keep = keep & (commit < receipt)
    keep = keep & (ship < commit)
    keep = keep & (receipt >= receipt_lo)
    keep = keep & (receipt < receipt_hi)
    
    # Load projection columns — only for rows that pass filter
    mode = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    prio = tl.load(priority_ptr + offs, mask=keep, other=0)
    
    # Compute per-group sums
    # For group 0 (MAIL, shipmode=0): sum of high_flag and low_flag
    high0_sum = tl.sum(tl.where(keep & (mode == 0), prio.to(tl.int64), 0))
    low0_sum = tl.sum(tl.where(keep & (mode == 0), (1 - prio.to(tl.int64)), 0))
    
    # For group 1 (SHIP, shipmode=1): sum of high_flag and low_flag  
    high1_sum = tl.sum(tl.where(keep & (mode == 1), prio.to(tl.int64), 0))
    low1_sum = tl.sum(tl.where(keep & (mode == 1), (1 - prio.to(tl.int64)), 0))
    
    # Atomic add to global outputs
    tl.atomic_add(out_high_ptr, high0_sum)
    tl.atomic_add(out_high_ptr + 1, high1_sum)
    tl.atomic_add(out_low_ptr, low0_sum)
    tl.atomic_add(out_low_ptr + 1, low1_sum)
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
    SQL: SELECT l_shipmode, 
         SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
         SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
         FROM orders, lineitem 
         WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL','SHIP')
         AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
         AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01'
         GROUP BY l_shipmode ORDER BY l_shipmode;
    
    Description: TPC-H Q12 - counts urgent vs non-urgent orders by ship mode.
    This implementation uses:
    1. cuDF for the join and string encoding (shipmode, order priority)
    2. A single fused Triton kernel for all filters, projection, and group-by aggregation
       - Reduces HBM traffic by loading filter columns first, conditionally loading projections
       - Performs group-by with atomic adds for two fixed groups
       - The kernel autotunes BLOCK_SIZE, num_warps, and num_stages for optimal performance
    
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

    # Convert dates to int32 epoch days
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

    n = shipmode_t.numel()
    device = shipmode_t.device
    
    # Output buffers: [mail_high, ship_high] and [mail_low, ship_low]
    high_out = torch.zeros(2, dtype=torch.int64, device=device)
    low_out = torch.zeros(2, dtype=torch.int64, device=device)

    # Launch fused Triton kernel
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q12_kernel[grid](
        shipmode_t, priority_t,
        l_commit_t, l_receipt_t, l_ship_t,
        receipt_lo_days, receipt_hi_days,
        high_out, low_out, n,
    )

    # Decode group ids back to strings; order alphabetically (MAIL < SHIP)
    names = [shipmode1, shipmode2]
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_out.cpu().tolist()
    lc_host = low_out.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count": [lc_host[i] for i in ordered],
    })
    return result
