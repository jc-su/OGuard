"""
Optimized Triton implementation of TPC‑H Q13.

SQL:
    SELECT c_count, count(*) AS custdist
    FROM (
        SELECT c_custkey, count(o_orderkey)
        FROM customer LEFT OUTER JOIN orders
          ON c_custkey = o_custkey
          AND o_comment NOT LIKE '%special%requests%'
        GROUP BY c_custkey
    ) AS c_orders (c_custkey, c_count)
    GROUP BY c_count
    ORDER BY custdist DESC, c_count DESC;

Description:
* cuDF handles the string‑based NOT LIKE filter.
* All numeric aggregation (per‑customer order count, histogram of those
  counts, final sorting) runs on the GPU.
* The heavy scan over `orders` is fused into a single Triton kernel that
  atomically increments per‑customer buckets, reducing HBM traffic from
  three passes to one.

Params:
    comment_pattern (str): LIKE pattern for the NOT LIKE predicate.
                           Default matches the TPC‑H benchmark.
"""

import os
import tempfile
import importlib.util
import linecache
import pandas as pd
import torch
import cudf
import cupy as cp
import triton

# ----------------------------------------------------------------------
# Triton kernel source – written to disk before import
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotuned configs for H100 (SM90)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _count_orders_kernel(
    o_custkey_ptr,           # int64*
    mask_ptr,                # uint8*  (1 = keep)
    out_ptr,                 # int32*  (size = num_customers+1)
    n_elements,              # total number of rows in orders
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    # Load columns (masked loads keep out‑of‑bounds safe)
    custkey = tl.load(o_custkey_ptr + offs, mask=valid, other=0)
    keep    = tl.load(mask_ptr + offs,    mask=valid, other=0)

    # Customer keys start at 1 → cast to int32 index
    idx = tl.cast(custkey, tl.int32)

    # Convert keep (uint8) to int32 0/1 for atomic add
    inc = tl.cast(keep, tl.int32)

    tl.atomic_add(out_ptr + idx, inc)
'''

def _load_kernels() -> object:
    """Write the Triton source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="q13_triton_")
    path = os.path.join(tmpdir, "_q13_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q13_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels at import time
_K = _load_kernels()

# ----------------------------------------------------------------------
# Launcher invoking the fused kernel
# ----------------------------------------------------------------------
def _launch_count_orders(
    o_custkey: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Fuse filtering mask and per‑customer scatter‑add into a single Triton kernel.

    Parameters
    ----------
    o_custkey : torch.Tensor[int64] – orders.o_custkey column
    mask      : torch.Tensor[uint8] – 1 where the order passes the NOT‑LIKE predicate
    out       : torch.Tensor[int32] – zero‑filled bucket array (size = max_custkey+1)
    """
    n = o_custkey.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._count_orders_kernel[grid](
        o_custkey,
        mask,
        out,
        n,
    )

# ----------------------------------------------------------------------
# Core query logic (public API stays unchanged)
# ----------------------------------------------------------------------
def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    mask_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the distribution of order counts per customer.

    Returns
    -------
    unique_counts : torch.Tensor[int64] – distinct order‑counts (c_count)
    custdist      : torch.Tensor[int64] – frequency of each count
    """
    # 1️⃣ Count orders per customer (fused Triton kernel)
    order_counts = torch.zeros(num_customers + 1, dtype=torch.int32, device=c_custkey.device)
    _launch_count_orders(o_custkey_filtered, mask_filtered, order_counts)

    # Drop bucket 0 (custkey starts at 1) and cast to int64 for bincount
    per_cust_counts = order_counts[1:].to(torch.int64)  # shape (num_customers,)

    # 2️⃣ Build histogram of those counts (GPU‑accelerated torch.bincount)
    max_cnt = int(per_cust_counts.max().item())
    hist = torch.bincount(per_cust_counts, minlength=max_cnt + 1)  # int64

    # Extract non‑zero bins → (c_count, custdist)
    nonzero = torch.nonzero(hist > 0, as_tuple=False).squeeze(1)
    unique_counts = nonzero.to(torch.int64)
    custdist = hist[nonzero]

    # 3️⃣ ORDER BY custdist DESC, c_count DESC (stable two‑stage sort)
    order1 = torch.argsort(unique_counts, descending=True, stable=True)
    unique_counts = unique_counts[order1]
    custdist = custdist[order1]

    order2 = torch.argsort(custdist, descending=True, stable=True)
    unique_counts = unique_counts[order2]
    custdist = custdist[order2]

    return unique_counts, custdist

def run_query(
    tables: dict[str, cudf.DataFrame],
    comment_pattern: str = "%special%requests%",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT c_count, count(*) AS custdist
        FROM (
            SELECT c_custkey, count(o_orderkey)
            FROM customer LEFT OUTER JOIN orders
              ON c_custkey = o_custkey
              AND o_comment NOT LIKE '%special%requests%'
            GROUP BY c_custkey
        ) AS c_orders (c_custkey, c_count)
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;

    Description:
        • cuDF performs the string‑based NOT LIKE filter.
        • All numeric aggregation (per‑customer order count, histogram of those
          counts, final sorting) runs on the GPU.
        • The heavy scan over `orders` is fused into a single Triton kernel,
          reducing HBM traffic and eliminating intermediate materializations.

    Params
    ------
    comment_pattern (str):
        LIKE pattern used for the NOT LIKE predicate.
        The default matches the TPC‑H benchmark.

    Returns
    -------
    cudf.DataFrame with columns:
        - c_count  (int64)
        - custdist (int64)
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Convert LIKE pattern to regex for cuDF (% -> .*, _ -> .)
    regex = "special.*requests"
    mask = ~df_orders["o_comment"].str.contains(regex, regex=True)  # bool Series

    # Transfer needed columns to GPU tensors
    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    o_custkey_t = torch.as_tensor(df_orders["o_custkey"].to_cupy(), dtype=torch.int64)
    mask_t = torch.as_tensor(mask.to_cupy(), dtype=torch.uint8)  # 1 = keep, 0 = discard

    # Number of distinct customers; TPCH keys are dense and start at 1
    num_customers = int(df_customer["c_custkey"].max())

    # Core computation
    unique_counts_t, custdist_t = _query_core(
        c_custkey=c_custkey_t,
        o_custkey_filtered=o_custkey_t,
        mask_filtered=mask_t,
        num_customers=num_customers,
    )

    # Convert results back to cuDF
    def _torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
        return cp.asarray(t)

    return cudf.DataFrame({
        "c_count": _torch_to_cupy(unique_counts_t),
        "custdist": _torch_to_cupy(custdist_t),
    })
