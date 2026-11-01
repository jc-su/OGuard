import os
import tempfile
import importlib.util
import linecache

import torch
import cudf
import triton

# ----------------------------------------------------------------------
# Triton kernel (saved to a temporary file and imported as a module)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Scatter‑add each order's customer key into a per‑customer counter.
# This kernel uses a fixed BLOCK_SIZE (8192) and no autotuning to
# avoid the autotune benchmarking overhead on the first launch.
@triton.jit
def _scatter_counts_kernel(
    custkey_ptr,   # int64*  (orders' o_custkey)
    out_ptr,       # int32*  (per‑customer counters)
    n_elements,    # total number of orders
    max_key,       # highest valid customer key (num_customers)
    BLOCK_SIZE: tl.constexpr = 8192,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offs < n_elements
    # Load a customer key; for OOB lanes load 0 (invalid)
    key = tl.load(custkey_ptr + offs, mask=mask, other=0)

    # Valid keys are in the range (0, max_key]
    valid = mask & (key > 0) & (key <= max_key)

    # Increment the bucket by 1 for valid rows; 0 otherwise.
    incr = tl.where(valid, 1, 0)
    tl.atomic_add(out_ptr + key, incr)
'''

def _load_kernels():
    """Write the Triton source to a temporary file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q13_triton_")
    path = os.path.join(tmpdir, "_q13_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q13_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels once at import time
_K = _load_kernels()

def _launch_scatter_counts(o_custkey_filtered: torch.Tensor,
                          num_customers: int) -> torch.Tensor:
    """
    Fuse the scatter‑add from the reference implementation into a single
    Triton kernel. Returns an ``int32`` tensor of length ``num_customers + 1``
    where index i holds the number of (filtered) orders for customer i.
    """
    n = o_custkey_filtered.numel()
    out = torch.zeros(num_customers + 1, dtype=torch.int32,
                      device=o_custkey_filtered.device)

    # One‑dimensional grid using the kernel's fixed BLOCK_SIZE (8192)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._scatter_counts_kernel[grid](
        o_custkey_filtered,
        out,
        n,
        num_customers,   # max_key
        # BLOCK_SIZE is defined inside the kernel; no need to pass it.
    )
    return out

def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core GPU compute for TPC‑H Q13.

    1. Count orders per customer via a fused Triton scatter‑add kernel.
    2. Gather each customer's order count (LEFT OUTER JOIN semantics → 0 if none).
    3. Compute the distribution of those counts (``c_count`` → ``custdist``).
    4. Order by ``custdist`` DESC, then ``c_count`` DESC (stable sorts).

    The heavy memory‑bound work (the scatter) is performed in a single
    pass with a fixed, pre‑selected block size to eliminate autotune overhead.
    """
    # Step 1 – fused scatter‑add (int32 counters)
    order_counts_int = _launch_scatter_counts(o_custkey_filtered, num_customers)

    # Step 2 – gather per‑customer order count (LEFT OUTER JOIN)
    # Cast customer keys to int32 for indexing into the int32 counter array.
    c_count = order_counts_int[c_custkey.to(torch.int32)]

    # Step 3 – distribution of counts
    unique_counts, custdist = torch.unique(c_count, return_counts=True)

    # Step 4 – ORDER BY custdist DESC, c_count DESC (stable sorts)
    order1 = torch.argsort(unique_counts, descending=True, stable=True)
    unique_counts = unique_counts[order1]
    custdist = custdist[order1]

    order2 = torch.argsort(custdist, descending=True, stable=True)
    unique_counts = unique_counts[order2]
    custdist = custdist[order2]

    return unique_counts, custdist

def run_query(tables: dict[str, cudf.DataFrame],
              comment_pattern: str = "%special%requests%") -> cudf.DataFrame:
    """
    SQL:
        SELECT c_count, count(*) AS custdist
        FROM (
            SELECT c_custkey, count(o_orderkey)
            FROM customer LEFT OUTER JOIN orders
              ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'
            GROUP BY c_custkey
        ) AS c_orders (c_custkey, c_count)
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;
    Description: TPC‑H Q13 – Customer distribution based on number of non‑special‑request orders.
    Params: comment_pattern (str): the LIKE pattern to exclude from orders via NOT LIKE.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Filter orders: o_comment NOT LIKE '%special%requests%'
    mask = ~df_orders["o_comment"].str.contains("special.*requests", regex=True)
    filtered_orders = df_orders[mask]

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(),
                                  dtype=torch.int64)
    o_custkey_t = torch.as_tensor(filtered_orders["o_custkey"].to_cupy(),
                                  dtype=torch.int64)

    # Upper bound for customer keys (bucket size)
    num_customers = int(df_customer["c_custkey"].max())

    c_count_t, custdist_t = _query_core(c_custkey_t, o_custkey_t,
                                        num_customers)

    return cudf.DataFrame({
        "c_count": c_count_t,
        "custdist": custdist_t,
    })
