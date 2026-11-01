import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

# -----------------------------------------------------------------------------
# Triton kernel source: fused priority counting kernel.
# -----------------------------------------------------------------------------
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
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _count_priority_kernel(priority_ptr, out_ptr, n_elements, num_priorities, BLOCK_SIZE: tl.constexpr):
    """Count occurrences of each priority code in a block and atomically add to global counts."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Use num_priorities as sentinel for masked-out positions so they don't
    # get counted as priority 0 (which could be a valid priority code).
    priorities = tl.load(priority_ptr + offs, mask=mask, other=num_priorities)

    # Per‑priority block reduction: for each priority value compute the
    # number of matching elements in this block and atomically add to the
    # global histogram.
    for p in range(num_priorities):
        block_count = tl.sum(tl.where(priorities == p, 1, 0))
        tl.atomic_add(out_ptr + p, block_count)
'''

def _load_kernels():
    """Load the Triton kernels from the on‑disk source."""
    tmpdir = tempfile.mkdtemp(prefix="q4_triton_")
    path = os.path.join(tmpdir, "_q4_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q4_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def _launch_count_priority(priority: torch.Tensor, num_priorities: int) -> torch.Tensor:
    """Launch the fused Triton kernel to count priority occurrences.

    Replaces the PyTorch `bincount` call with a kernel that performs a
    block‑wise reduction and a single atomic add per priority, reducing
    atomic contention and HBM traffic.
    """
    n = priority.numel()
    counts = torch.zeros(num_priorities, dtype=torch.int64, device=priority.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._count_priority_kernel[grid](priority, counts, n, num_priorities)
    return counts


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
    """Hot‑path for TPC‑H Q4: filter orders, semi‑join with lineitem, and aggregate by priority.

    The final group‑by count is offloaded to a Triton kernel to fuse the reduction
    with the loading of priority codes, reducing memory traffic compared to
    `torch.bincount`.
    """
    # Filter orders by date range
    order_mask = (o_orderdate_days >= date_lo_days) & (o_orderdate_days < date_hi_days)
    filt_orderkey = o_orderkey[order_mask]
    filt_priority = o_orderpriority_codes[order_mask]

    # Filter lineitem rows where commitdate < receiptdate
    line_mask = l_commitdate_days < l_receiptdate_days
    qualifying_l_orderkey = l_orderkey[line_mask]

    # EXISTS: check which filt_orderkey values appear in qualifying_l_orderkey
    unique_keys = torch.unique(qualifying_l_orderkey)
    exists_mask = torch.isin(filt_orderkey, unique_keys)

    final_priority = filt_priority[exists_mask]

    # Count priorities using the fused Triton kernel
    counts = _launch_count_priority(final_priority, num_priorities)
    priority_ids = torch.arange(num_priorities, device=counts.device, dtype=torch.int64)

    # Keep only non‑zero groups
    nz = counts > 0
    return priority_ids[nz], counts[nz]


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
