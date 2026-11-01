import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source – all @triton.jit/@triton.autotune definitions here.
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=tw, num_stages=ns)
        for bs in [1024, 2048, 4096, 8192]
        for tw in [4, 8]
        for ns in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_high_ptr", "out_low_ptr"],
)
@triton.jit
def _q12_kernel(
    shipmode_ptr,            # int32 *
    high_flag_ptr,           # int32 *
    commit_ptr,              # int32 *
    receipt_ptr,             # int32 *
    shipdate_ptr,            # int32 *
    receipt_lo: tl.int32,    # scalar bound
    receipt_hi: tl.int32,    # scalar bound
    out_high_ptr,            # int64 *
    out_low_ptr,             # int64 *
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load the three date columns (masked)
    commit   = tl.load(commit_ptr   + offs, mask=mask,    other=0)
    receipt  = tl.load(receipt_ptr  + offs, mask=mask,    other=0)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask,    other=0)

    # Predicate evaluation (int32)
    pred = (commit < receipt) & (shipdate < commit) & \
           (receipt >= receipt_lo) & (receipt < receipt_hi)

    keep = mask & pred

    # Conditional loads for shipmode and high flag (masked)
    shipmode = tl.load(shipmode_ptr + offs, mask=keep, other=0)
    high_flag = tl.load(high_flag_ptr + offs, mask=keep, other=0)

    # Convert flags to int64 while respecting the mask
    high_i = tl.where(keep, tl.cast(high_flag, tl.int64), 0)
    low_i  = tl.where(keep, tl.cast(1 - high_flag, tl.int64), 0)

    # Atomic adds per shipmode (0 = MAIL, 1 = SHIP)
    tl.atomic_add(out_high_ptr + shipmode, high_i)
    tl.atomic_add(out_low_ptr  + shipmode, low_i)
'''
def _load_kernels():
    """Write the Triton source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q12_triton_")
    path = os.path.join(tmpdir, "_q12_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q12_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels once at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Private launcher – uses Triton's autotuned BLOCK_SIZE (no explicit BLOCK_SIZE argument).
# ----------------------------------------------------------------------
def _launch_q12(
    shipmode: torch.Tensor,
    high_flag: torch.Tensor,
    commit: torch.Tensor,
    receipt: torch.Tensor,
    shipdate: torch.Tensor,
    receipt_lo: int,
    receipt_hi: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the fused kernel and return per‑group high/low counts as int64 tensors."""
    n = shipmode.numel()
    out_high = torch.zeros(2, dtype=torch.int64, device=shipmode.device)
    out_low  = torch.zeros(2, dtype=torch.int64, device=shipmode.device)

    # Triton determines the optimal BLOCK_SIZE via autotune
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._q12_kernel[grid](
        shipmode,
        high_flag,
        commit,
        receipt,
        shipdate,
        receipt_lo,
        receipt_hi,
        out_high,
        out_low,
        n,
    )
    return out_high, out_low

# ----------------------------------------------------------------------
# Public core function – signature must stay identical to the reference.
# ----------------------------------------------------------------------
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
    Fuse filter, projection, and two‑group aggregation into a single Triton kernel.
    This reduces memory traffic to one pass over the data and uses int64 atomics,
    which are faster than the float64 atomics used previously.
    """
    high_counts, low_counts = _launch_q12(
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

# ----------------------------------------------------------------------
# Public API – unchanged from the reference implementation.
# ----------------------------------------------------------------------
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

    # Pre‑filter lineitem on shipmode to shrink the join size
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
