import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused kernel for counting orders per customer (scatter add)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ----------------------------------------------------------------------
// Kernel: atomic increment each bucket corresponding to a customer key
// ----------------------------------------------------------------------
__global__ void scatter_counts_kernel(
    const int64_t* __restrict__ keys,
    int32_t* __restrict__ out,
    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Customer keys are 1‑based, fit in int32 for TPCH scale 10
        int32_t key = static_cast<int32_t>(keys[idx]);
        // Simple atomic add – one per matching order
        atomicAdd(&out[key], 1);
    }
}

// ----------------------------------------------------------------------
// Launcher callable from Python (via pybind11)
// ----------------------------------------------------------------------
torch::Tensor scatter_counts_launcher(torch::Tensor keys, int64_t num_customers) {
    TORCH_CHECK(keys.is_cuda(), "keys must be a CUDA tensor");
    TORCH_CHECK(keys.dtype() == torch::kInt64, "keys must be int64");
    TORCH_CHECK(keys.is_contiguous(), "keys must be contiguous");
    int64_t n = keys.numel();

    // Output: int32 histogram, index 0 unused (custkeys start at 1)
    auto options = torch::TensorOptions()
                       .dtype(torch::kInt32)
                       .device(keys.device());
    torch::Tensor out = torch::zeros({num_customers + 1}, options);

    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    scatter_counts_kernel<<<blocks, threads>>>(
        keys.data_ptr<int64_t>(),
        out.data_ptr<int32_t>(),
        n);

    // Ensure kernel completion before returning to Python
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("scatter_counts_kernel launch failed: ", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        AT_ERROR("scatter_counts_kernel sync failed: ", cudaGetErrorString(err));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_counts_launcher",
          &scatter_counts_launcher,
          "Scatter‑add order keys into per‑customer counts (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Helper: compile the CUDA source once at import time
# ----------------------------------------------------------------------
def _load_cuda_extension() -> object:
    """Compile the embedded CUDA source and return the loaded module."""
    tmpdir = tempfile.mkdtemp(prefix="q13_cuda_")
    cu_path = os.path.join(tmpdir, "q13_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)

    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q13_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        verbose=False,
        build_directory=build_dir,
        with_cuda=True,
    )


# Load (or re‑load) the extension exactly once.
_EXT = _load_cuda_extension()


# ----------------------------------------------------------------------
# Private launcher that wraps the compiled CUDA function
# ----------------------------------------------------------------------
def _launch_scatter_counts(o_custkey_filtered: torch.Tensor, num_customers: int) -> torch.Tensor:
    """
    Invoke the CUDA kernel that builds ``order_counts_per_cust``.
    Returns an ``int32`` tensor of shape ``(num_customers + 1,)`` on the
    same device as ``o_custkey_filtered``.
    """
    # Ensure proper dtype / device / contiguity for the kernel.
    keys = o_custkey_filtered.contiguous()
    if not keys.is_cuda:
        keys = keys.cuda()
    return _EXT.scatter_counts_launcher(keys, num_customers)


# ----------------------------------------------------------------------
# Public core function – signature fixed by the level constraints
# ----------------------------------------------------------------------
def _query_core(
    c_custkey: torch.Tensor,
    o_custkey_filtered: torch.Tensor,
    num_customers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU‑accelerated computation of Q13's aggregation.

    Steps:
    1. **Scatter‑add** ``o_custkey_filtered`` into a per‑customer count array
       using a custom CUDA kernel (fused, coalesced, single HBM pass).
    2. Gather the count for each customer key (LEFT OUTER JOIN semantics).
    3. ``torch.unique`` with ``return_counts=True`` to obtain
       ``c_count`` histogram (``custdist``).
    4. Two-stage stable sorting to match the reference ordering:
       first by ``c_count`` descending, then by ``custdist`` descending.
    """
    # 1. Build order_counts_per_cust[0..num_customers] (int32)
    order_counts_int32 = _launch_scatter_counts(o_custkey_filtered, num_customers)

    # Convert to int64 for downstream indexing (torch scatter expects int64)
    order_counts = order_counts_int32.to(torch.int64)

    # 2. Gather per‑customer order count; keys are 1‑based.
    c_counts = order_counts.gather(0, c_custkey.to(torch.int64))

    # 3. Histogram of counts: unique c_counts and their frequencies.
    unique_counts, custdist = torch.unique(c_counts, return_counts=True)

    # 4. Ordering – same as reference plan.
    order1 = torch.argsort(unique_counts, descending=True, stable=True)
    unique_counts = unique_counts[order1]
    custdist = custdist[order1]

    order2 = torch.argsort(custdist, descending=True, stable=True)
    unique_counts = unique_counts[order2]
    custdist = custdist[order2]

    return unique_counts, custdist


# ----------------------------------------------------------------------
# Public entry point – verbatim from the reference plan (unchanged)
# ----------------------------------------------------------------------
def run_query(tables: dict[str, cudf.DataFrame], comment_pattern: str = "%special%requests%") -> cudf.DataFrame:
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

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    o_custkey_t = torch.as_tensor(filtered_orders["o_custkey"].to_cupy(), dtype=torch.int64)

    # Use max custkey as the bucket size upper bound
    num_customers = int(df_customer["c_custkey"].max())

    c_count_t, custdist_t = _query_core(c_custkey_t, o_custkey_t, num_customers)

    return cudf.DataFrame({
        "c_count": c_count_t,
        "custdist": custdist_t,
    })
