import os
import tempfile

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused kernels for TPC‑H Q4
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// ---------------------------------------------------------------
// Kernel 1: mark order keys that have at least one lineitem with
//           commitdate < receiptdate.
// ---------------------------------------------------------------
__global__ void lineitem_flag_kernel(
    const int64_t* __restrict__ l_orderkey,
    const int32_t* __restrict__ l_commit,
    const int32_t* __restrict__ l_receipt,
    int32_t* __restrict__ flags,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (l_commit[idx] < l_receipt[idx]) {
        int64_t key = l_orderkey[idx];
        // set flag[key] = 1 (atomic OR with 1)
        atomicOr(&flags[key], 1);
    }
}

// ---------------------------------------------------------------
// Kernel 2: for each order, if its date is in range and its key
//           is flagged, increment the per‑priority counter.
// ---------------------------------------------------------------
__global__ void order_agg_kernel(
    const int64_t* __restrict__ o_orderkey,
    const int32_t* __restrict__ o_orderdate,
    const int32_t* __restrict__ o_priority,
    const int32_t* __restrict__ flags,
    int32_t date_lo,
    int32_t date_hi,
    int64_t* __restrict__ counts,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t d = o_orderdate[idx];
    if (d >= date_lo && d < date_hi) {
        int64_t key = o_orderkey[idx];
        if (flags[key] != 0) {
            int32_t p = o_priority[idx];
            // atomicAdd on int64_t via unsigned long long cast (supported on sm_90)
            atomicAdd(reinterpret_cast<unsigned long long*>(&counts[p]), 1ULL);
        }
    }
}

// ---------------------------------------------------------------
// Launcher: allocate flag buffer, run both kernels, return counts.
// ---------------------------------------------------------------
torch::Tensor q4_launcher(
    torch::Tensor o_orderkey,
    torch::Tensor o_orderdate,
    torch::Tensor o_priority,
    torch::Tensor l_orderkey,
    torch::Tensor l_commit,
    torch::Tensor l_receipt,
    int32_t date_lo,
    int32_t date_hi,
    int64_t num_priorities)
{
    // Basic sanity (lightweight – heavy checks done in Python)
    TORCH_CHECK(o_orderkey.is_cuda() && o_orderkey.dtype() == torch::kInt64);
    TORCH_CHECK(o_orderdate.is_cuda() && o_orderdate.dtype() == torch::kInt32);
    TORCH_CHECK(o_priority.is_cuda() && o_priority.dtype() == torch::kInt32);
    TORCH_CHECK(l_orderkey.is_cuda() && l_orderkey.dtype() == torch::kInt64);
    TORCH_CHECK(l_commit.is_cuda() && l_commit.dtype() == torch::kInt32);
    TORCH_CHECK(l_receipt.is_cuda() && l_receipt.dtype() == torch::kInt32);

    // Ensure contiguous layout
    o_orderkey = o_orderkey.contiguous();
    o_orderdate = o_orderdate.contiguous();
    o_priority = o_priority.contiguous();
    l_orderkey = l_orderkey.contiguous();
    l_commit = l_commit.contiguous();
    l_receipt = l_receipt.contiguous();

    // Determine size for flag buffer (max key seen in either table)
    int64_t max_key_o = o_orderkey.max().item<int64_t>();
    int64_t max_key_l = l_orderkey.max().item<int64_t>();
    int64_t max_key   = max_key_o > max_key_l ? max_key_o : max_key_l;

    auto opt_int32 = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .device(o_orderkey.device());
    torch::Tensor flags = torch::zeros({max_key + 1}, opt_int32);

    // -----------------------------------------------------------
    // Launch lineitem flag kernel
    // -----------------------------------------------------------
    int64_t n_line = l_orderkey.numel();
    int blocks_line = (n_line + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lineitem_flag_kernel<<<blocks_line, BLOCK_SIZE>>>(
        l_orderkey.data_ptr<int64_t>(),
        l_commit.data_ptr<int32_t>(),
        l_receipt.data_ptr<int32_t>(),
        flags.data_ptr<int32_t>(),
        n_line
    );

    // -----------------------------------------------------------
    // Launch order aggregation kernel
    // -----------------------------------------------------------
    int64_t n_order = o_orderkey.numel();
    int blocks_order = (n_order + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto opt_int64 = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .device(o_orderkey.device());
    torch::Tensor counts = torch::zeros({num_priorities}, opt_int64);

    order_agg_kernel<<<blocks_order, BLOCK_SIZE>>>(
        o_orderkey.data_ptr<int64_t>(),
        o_orderdate.data_ptr<int32_t>(),
        o_priority.data_ptr<int32_t>(),
        flags.data_ptr<int32_t>(),
        date_lo,
        date_hi,
        counts.data_ptr<int64_t>(),
        n_order
    );

    // Implicit stream synchronization: kernels are queued in order.
    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q4_launcher", &q4_launcher, "TPC‑H Q4 fused CUDA launcher");
}
'''


def _load_cuda():
    """Compile the embedded CUDA source and return the loaded extension."""
    tmpdir = tempfile.mkdtemp(prefix="q4_cuda_")
    cu_path = os.path.join(tmpdir, "q4_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q4_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


# Compile once at import time.
_EXT = _load_cuda()


def _launch_q4_counts(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_priority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
) -> torch.Tensor:
    """
    Execute the fused CUDA kernel that returns a 1‑D int64 tensor
    ``counts`` where ``counts[p]`` equals the number of qualifying orders
    with priority code ``p``.
    """
    device = o_orderkey.device
    # Ensure all tensors are contiguous and on the same CUDA device.
    o_orderkey = o_orderkey.contiguous().to(device)
    o_orderdate_days = o_orderdate_days.contiguous().to(device)
    o_priority_codes = o_priority_codes.contiguous().to(device)
    l_orderkey = l_orderkey.contiguous().to(device)
    l_commitdate_days = l_commitdate_days.contiguous().to(device)
    l_receiptdate_days = l_receiptdate_days.contiguous().to(device)

    counts = _EXT.q4_launcher(
        o_orderkey,
        o_orderdate_days,
        o_priority_codes,
        l_orderkey,
        l_commitdate_days,
        l_receiptdate_days,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )
    return counts  # already a torch.Tensor on the correct device


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
    Core hot‑path replacement: a fused CUDA implementation that
    1. flags order keys present in lineitem rows satisfying
       ``l_commitdate < l_receiptdate``,
    2. scans the filtered orders (date range) and increments a per‑priority
       counter when the order key is flagged.

    Returns the same tuple as the reference implementation:
    ``(priority_ids, counts)`` where both are 1‑D tensors of equal length,
    containing only the non‑zero groups, already sorted by priority code.
    """
    counts = _launch_q4_counts(
        o_orderkey,
        o_orderdate_days,
        o_orderpriority_codes,
        l_orderkey,
        l_commitdate_days,
        l_receiptdate_days,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )
    # Build the id tensor and drop zero‑count groups (identical to reference).
    priority_ids = torch.arange(num_priorities, device=counts.device, dtype=torch.int64)
    nz = counts > 0
    return priority_ids[nz], counts[nz]


def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count FROM orders
    WHERE o_orderdate >= CAST('1993-07-01' AS date)
      AND o_orderdate < CAST('1993-10-01' AS date)
      AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey
                  AND l_commitdate < l_receiptdate)
    GROUP BY o_orderpriority ORDER BY o_orderpriority;
    Description: TPC‑H Q4 – counts orders per priority within a date range where
    at least one lineitem shipped late (commit before receipt).
    Params:
        date_lo (str): inclusive lower bound for o_orderdate.
        date_hi (str): exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    # Encode o_orderpriority as categorical for group‑by on GPU
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

    # Map codes back to priority strings and sort (already sorted by code)
    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[i] for i in priority_ids_host]

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
        "order_count": counts_t,
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
