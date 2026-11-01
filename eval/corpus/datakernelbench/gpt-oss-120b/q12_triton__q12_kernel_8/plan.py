import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source (written to a temporary file at import time)
# ----------------------------------------------------------------------
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
    shipmode_ptr,      # int32 *
    priority_ptr,      # int32 *
    commit_ptr,        # int32 *
    receipt_ptr,       # int32 *
    ship_ptr,          # int32 *
    receipt_lo,        # int32 scalar
    receipt_hi,        # int32 scalar
    out_high_ptr,      # int64 *
    out_low_ptr,       # int64 *
    n_elements,        # int64 scalar
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load date columns (masked)
    commit = tl.load(commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=0)
    ship = tl.load(ship_ptr + offs, mask=mask, other=0)

    # Date predicates
    keep = (
        mask
        & (commit < receipt)
        & (ship < commit)
        & (receipt >= receipt_lo)
        & (receipt < receipt_hi)
    )

    # Load cheap columns only for rows that satisfy date predicates
    shipmode = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    priority = tl.load(priority_ptr + offs, mask=keep, other=0)   # 1 = high, 0 = low

    # Per‑lane contributions (int64)
    high0 = tl.where(keep & (shipmode == 0) & (priority == 1), 1, 0).to(tl.int64)
    high1 = tl.where(keep & (shipmode == 1) & (priority == 1), 1, 0).to(tl.int64)
    low0  = tl.where(keep & (shipmode == 0) & (priority == 0), 1, 0).to(tl.int64)
    low1  = tl.where(keep & (shipmode == 1) & (priority == 0), 1, 0).to(tl.int64)

    # Block‑wide reductions
    s_high0 = tl.sum(high0)
    s_high1 = tl.sum(high1)
    s_low0  = tl.sum(low0)
    s_low1  = tl.sum(low1)

    # One atomic add per block per group
    tl.atomic_add(out_high_ptr + 0, s_high0)
    tl.atomic_add(out_high_ptr + 1, s_high1)
    tl.atomic_add(out_low_ptr + 0,  s_low0)
    tl.atomic_add(out_low_ptr + 1,  s_low1)
'''

def _load_kernels():
    """Write the kernel source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="q12_triton_")
    path = os.path.join(tmpdir, "_q12_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q12_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load Triton kernels once at import time
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
    Run the fused Triton kernel that:
      * applies all date predicates,
      * groups by shipmode (0 = shipmode1, 1 = shipmode2),
      * counts high‑priority and low‑priority orders per shipmode.
    Returns (group_ids, high_counts, low_counts) matching the reference API.
    """
    n = l_shipmode_code.numel()
    out_high = torch.zeros(2, dtype=torch.int64, device=l_shipmode_code.device)
    out_low  = torch.zeros(2, dtype=torch.int64, device=l_shipmode_code.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q12_kernel[grid](
        l_shipmode_code,
        o_orderpriority_high,
        l_commitdate,
        l_receiptdate,
        l_shipdate,
        receipt_lo_days,
        receipt_hi_days,
        out_high,
        out_low,
        n,
    )
    group_ids = torch.arange(2, device=l_shipmode_code.device, dtype=torch.int64)
    return group_ids, out_high, out_low

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
               SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
               SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
        FROM   orders, lineitem
        WHERE  o_orderkey = l_orderkey
               AND l_shipmode IN ('MAIL','SHIP')
               AND l_commitdate < l_receiptdate
               AND l_shipdate < l_commitdate
               AND l_receiptdate >= CAST('1994-01-01' AS date)
               AND l_receiptdate <  CAST('1995-01-01' AS date)
        GROUP BY l_shipmode
        ORDER BY l_shipmode;

    Description:
        The join and shipmode filter are performed with cuDF.  All per‑row
        predicates, simple projections, and the two tiny GROUP‑BY
        aggregations (high and low counts) are fused into a single Triton
        kernel (`_q12_kernel`).  This eliminates intermediate materialisation,
        reduces HBM traffic, and aggregates via block‑wise reductions,
        delivering a measurable speedup on the H100.

    Params:
        shipmode1, shipmode2 – allowed shipping modes.
        receipt_lo, receipt_hi – inclusive lower / exclusive upper bounds for l_receiptdate.
        priority1, priority2 – order‑priority values counted as “high”.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Convert date bounds to int32 epoch‑day representation
    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    # Pre‑filter lineitem on shipmode to reduce join size
    li = df_lineitem[df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])]

    # Perform inner join
    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey")

    # Encode shipmode as 0/1 (MAIL=0, SHIP=1)
    shipmode_code = (joined["l_shipmode"] == shipmode2).astype("int32")

    # Encode high‑priority flag (1 = high, 0 = low)
    priority_high = (
        (joined["o_orderpriority"] == priority1) |
        (joined["o_orderpriority"] == priority2)
    ).astype("int32")

    # Convert date columns to int32 days since epoch
    commit_days = ((joined["l_commitdate"].astype("int64") // 86400).astype("int32"))
    receipt_days = ((joined["l_receiptdate"].astype("int64") // 86400).astype("int32"))
    ship_days = ((joined["l_shipdate"].astype("int64") // 86400).astype("int32"))

    # Transfer to Torch tensors on GPU (via CuPy bridge)
    shipmode_t = torch.as_tensor(shipmode_code.to_cupy(), dtype=torch.int32)
    priority_t = torch.as_tensor(priority_high.to_cupy(), dtype=torch.int32)
    commit_t   = torch.as_tensor(commit_days.to_cupy(), dtype=torch.int32)
    receipt_t  = torch.as_tensor(receipt_days.to_cupy(), dtype=torch.int32)
    ship_t     = torch.as_tensor(ship_days.to_cupy(), dtype=torch.int32)

    # Execute fused kernel
    group_ids, high_counts, low_counts = _query_core(
        shipmode_t,
        priority_t,
        commit_t,
        receipt_t,
        ship_t,
        receipt_lo_days,
        receipt_hi_days,
    )

    # Decode group IDs back to strings and order alphabetically (MAIL < SHIP)
    names = [shipmode1, shipmode2]            # index 0 -> shipmode1, 1 -> shipmode2
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_counts.cpu().tolist()
    lc_host = low_counts.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count": [lc_host[i] for i in ordered],
    })
    return result
