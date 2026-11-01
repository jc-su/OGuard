import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# Triton kernel source: fused filter + aggregate for Q12
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
    reset_to_zero=["high_ptr", "low_ptr"],
)
@triton.jit
def _q12_kernel(
    shipmode_ptr,
    priority_ptr,
    commit_ptr,
    receipt_ptr,
    ship_ptr,
    receipt_lo,
    receipt_hi,
    high_ptr,
    low_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns
    shipmode = tl.load(shipmode_ptr + offs, mask=mask, other=0)
    priority = tl.load(priority_ptr + offs, mask=mask, other=0)
    commit = tl.load(commit_ptr + offs, mask=mask, other=0)
    receipt = tl.load(receipt_ptr + offs, mask=mask, other=0)
    ship = tl.load(ship_ptr + offs, mask=mask, other=0)

    # Filter predicate
    keep = mask & (commit < receipt) & (ship < commit) & (receipt >= receipt_lo) & (receipt < receipt_hi)

    # Cast to int64 for accumulation
    priority_i64 = priority.to(tl.int64)

    # Contributions
    high_contrib = tl.where(keep, priority_i64, 0)
    low_contrib = tl.where(keep, 1 - priority_i64, 0)

    # Group 0 (MAIL)
    high_g0 = tl.sum(tl.where(shipmode == 0, high_contrib, 0))
    low_g0 = tl.sum(tl.where(shipmode == 0, low_contrib, 0))
    # Group 1 (SHIP)
    high_g1 = tl.sum(tl.where(shipmode == 1, high_contrib, 0))
    low_g1 = tl.sum(tl.where(shipmode == 1, low_contrib, 0))

    # Atomic add to global counters
    tl.atomic_add(high_ptr + 0, high_g0)
    tl.atomic_add(low_ptr + 0, low_g0)
    tl.atomic_add(high_ptr + 1, high_g1)
    tl.atomic_add(low_ptr + 1, low_g1)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q12_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _launch_q12(
    shipmode_t: torch.Tensor,
    priority_t: torch.Tensor,
    commit_t: torch.Tensor,
    receipt_t: torch.Tensor,
    ship_t: torch.Tensor,
    receipt_lo_days: int,
    receipt_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = shipmode_t.shape[0]
    high_counts = torch.zeros(2, dtype=torch.int64, device=shipmode_t.device)
    low_counts = torch.zeros(2, dtype=torch.int64, device=shipmode_t.device)
    if n == 0:
        return high_counts, low_counts
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q12_kernel[grid](
        shipmode_t,
        priority_t,
        commit_t,
        receipt_t,
        ship_t,
        receipt_lo_days,
        receipt_hi_days,
        high_counts,
        low_counts,
        n,
    )
    return high_counts, low_counts


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
        FROM orders, lineitem
        WHERE o_orderkey = l_orderkey
          AND l_shipmode IN ('MAIL','SHIP')
          AND l_commitdate < l_receiptdate
          AND l_shipdate < l_commitdate
          AND l_receiptdate >= CAST('1994-01-01' AS date)
          AND l_receiptdate < CAST('1995-01-01' AS date)
        GROUP BY l_shipmode
        ORDER BY l_shipmode;

    Description:
        TPC‑H Q12: count “high” (urgent or high priority) versus “low” line items
        per ship mode, applying the date constraints on receipt date.
        This implementation performs the join in cuDF, then runs a single fused
        Triton kernel that applies all filter predicates and aggregates
        high/low counts per ship mode in one pass on the GPU.

    Parameters
    ----------
    tables : dict[str, cudf.DataFrame]
        Dictionary containing the 'orders' and 'lineitem' DataFrames.
    shipmode1 : str, optional
        First shipping mode to consider (default 'MAIL').
    shipmode2 : str, optional
        Second shipping mode to consider (default 'SHIP').
    receipt_lo : str, optional
        Lower bound for receipt date (default '1994-01-01').
    receipt_hi : str, optional
        Upper bound for receipt date (default '1995-01-01').
    priority1 : str, optional
        First high‑priority order priority (default '1-URGENT').
    priority2 : str, optional
        Second high‑priority order priority (default '2-HIGH').

    Returns
    -------
    cudf.DataFrame
        DataFrame with columns ['l_shipmode', 'high_line_count', 'low_line_count'],
        ordered alphabetically by ship mode.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Convert date bounds to integer days since epoch
    receipt_lo_days = int(pd.to_datetime(receipt_lo).timestamp() // 86400)
    receipt_hi_days = int(pd.to_datetime(receipt_hi).timestamp() // 86400)

    # Pre‑filter lineitem on ship mode (reduces join cardinality)
    li = df_lineitem[df_lineitem["l_shipmode"].isin([shipmode1, shipmode2])]

    # Join with orders
    joined = li.merge(df_orders, left_on="l_orderkey", right_on="o_orderkey")

    # Encode ship mode: MAIL -> 0, SHIP -> 1
    shipmode_code = (joined["l_shipmode"] == shipmode2).astype("int32")

    # Encode high priority
    priority_high = (
        (joined["o_orderpriority"] == priority1) | (joined["o_orderpriority"] == priority2)
    ).astype("int32")

    # Convert date columns to int32 epoch days
    l_commit_t = torch.as_tensor(
        ((joined["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    l_receipt_t = torch.as_tensor(
        ((joined["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    l_ship_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )
    shipmode_t = torch.as_tensor(shipmode_code.to_cupy(), dtype=torch.int32)
    priority_t = torch.as_tensor(priority_high.to_cupy(), dtype=torch.int32)

    # Launch fused Triton kernel
    high_counts, low_counts = _launch_q12(
        shipmode_t,
        priority_t,
        l_commit_t,
        l_receipt_t,
        l_ship_t,
        receipt_lo_days,
        receipt_hi_days,
    )

    # Build result DataFrame, preserving alphabetical order of ship modes
    names = [shipmode1, shipmode2]
    ordered = sorted(range(2), key=lambda i: names[i])
    hc_host = high_counts.cpu().tolist()
    lc_host = low_counts.cpu().tolist()

    result = cudf.DataFrame({
        "l_shipmode": [names[i] for i in ordered],
        "high_line_count": [hc_host[i] for i in ordered],
        "low_line_count": [lc_host[i] for i in ordered],
    })
    return result
