import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

# ----------------------------------------------------------------------
# Triton kernel source (must be written to disk before import)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _bincount_kernel(
    priority_ptr,          # *int32  (order priority codes)
    n_elements,           # i64    (number of elements)
    out_ptr,              # *int64 (per‑priority counters)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load priority code (int32) and promote to int64 for pointer arithmetic
    pr = tl.load(priority_ptr + offs, mask=mask, other=0)
    pr = pr.to(tl.int64)

    # Increment by 1 where the lane is active, otherwise add 0 (no‑op)
    inc = tl.where(mask, 1, 0)
    tl.atomic_add(out_ptr + pr, inc)
'''

def _load_kernels():
    """Write the Triton source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q04_triton_")
    path = os.path.join(tmpdir, "_q04_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q04_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels at import time – they are ready before any query runs
_K = _load_kernels()

# ----------------------------------------------------------------------
# Helper: Triton‑based per‑priority bincount
# ----------------------------------------------------------------------
def _launch_bincount(
    priority: torch.Tensor,
    num_priorities: int,
) -> torch.Tensor:
    """
    Fused Triton kernel that computes a histogram of `priority` codes.
    Returns an int64 tensor of length `num_priorities` with the counts.
    """
    n = priority.numel()
    out = torch.zeros(num_priorities, dtype=torch.int64, device=priority.device)
    if n == 0:
        return out
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    # Triton will pick the best BLOCK_SIZE from the autotune configs
    _K._bincount_kernel[grid](
        priority,
        n,
        out,
    )
    return out

# ----------------------------------------------------------------------
# Core compute – unchanged signature, uses Triton for the aggregation
# ----------------------------------------------------------------------
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
    """
    Compute the TPC‑H Q4 result.
    The filtering and existence checks use regular PyTorch ops (memory‑bound but cheap),
    while the final per‑priority count is performed by a fused Triton kernel,
    reducing the number of HBM passes compared with ``torch.bincount``.
    """
    # 1️⃣ Filter orders by the requested date range
    order_mask = (o_orderdate_days >= date_lo_days) & (o_orderdate_days < date_hi_days)
    filt_orderkey = o_orderkey[order_mask]
    filt_priority = o_orderpriority_codes[order_mask]

    # 2️⃣ Filter lineitems where commitdate < receiptdate
    line_mask = l_commitdate_days < l_receiptdate_days
    qualifying_l_orderkey = l_orderkey[line_mask]

    # 3️⃣ Existence check: does a filtered order appear in qualifying lineitems?
    unique_keys = torch.unique(qualifying_l_orderkey)
    exists_mask = torch.isin(filt_orderkey, unique_keys)

    # 4️⃣ Keep priorities of orders that passed the EXISTS test
    final_priority = filt_priority[exists_mask]

    # 5️⃣ Triton‑fused per‑priority histogram (count)
    counts = _launch_bincount(final_priority, num_priorities)

    # 6️⃣ Build final result (only non‑zero groups, as reference does)
    priority_ids = torch.arange(num_priorities, device=counts.device, dtype=torch.int64)
    nz = counts > 0
    return priority_ids[nz], counts[nz]

# ----------------------------------------------------------------------
# Public API – unchanged from reference implementation
# ----------------------------------------------------------------------
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
