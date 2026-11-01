import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Fused kernel: compute value = supplycost * availqty and aggregate by inverse index
// Uses block-level shared memory reduction before atomicAdd to minimize global atomics
__global__ void q11_fused_aggregate_kernel(
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ availqty_ptr,
    const int64_t* __restrict__ inverse_ptr,
    double* __restrict__ agg_ptr,
    int64_t n_elements,
    int64_t n_unique_keys
) {
    extern __shared__ double shared_vals[];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory for this block's portion of unique keys
    // Each block handles a subset of the aggregation space
    for (int i = tid; i < n_unique_keys; i += blockDim.x) {
        shared_vals[i] = 0.0;
    }
    __syncthreads();
    
    // Process elements: compute value and accumulate in shared memory by inverse index
    if (idx < n_elements) {
        double value = supplycost_ptr[idx] * availqty_ptr[idx];
        int64_t key_idx = inverse_ptr[idx];
        // Atomic add to shared memory position for this key
        atomicAdd(&shared_vals[key_idx], value);
    }
    __syncthreads();
    
    // Reduce shared memory to global: each thread handles a subset of keys
    for (int i = tid; i < n_unique_keys; i += blockDim.x) {
        if (shared_vals[i] > 0.0) {
            atomicAdd(&agg_ptr[i], shared_vals[i]);
        }
    }
}

// Simpler kernel: direct atomic add per thread (fallback for small data)
__global__ void q11_simple_aggregate_kernel(
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ availqty_ptr,
    const int64_t* __restrict__ inverse_ptr,
    double* __restrict__ agg_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        double value = supplycost_ptr[idx] * availqty_ptr[idx];
        int64_t key_idx = inverse_ptr[idx];
        atomicAdd(&agg_ptr[key_idx], value);
    }
}

// Launcher for fused aggregation
torch::Tensor q11_aggregate_launcher(
    torch::Tensor supplycost,
    torch::Tensor availqty,
    torch::Tensor inverse,
    int64_t n_unique_keys
) {
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64, "supplycost must be float64");
    TORCH_CHECK(availqty.scalar_type() == torch::kFloat64, "availqty must be float64");
    TORCH_CHECK(inverse.scalar_type() == torch::kInt64, "inverse must be int64");
    TORCH_CHECK(supplycost.is_cuda(), "supplycost must be on CUDA");
    TORCH_CHECK(availqty.is_cuda(), "availqty must be on CUDA");
    TORCH_CHECK(inverse.is_cuda(), "inverse must be on CUDA");
    
    supplycost = supplycost.contiguous();
    availqty = availqty.contiguous();
    inverse = inverse.contiguous();
    
    int64_t n_elements = supplycost.numel();
    TORCH_CHECK(availqty.numel() == n_elements, "availqty size mismatch");
    TORCH_CHECK(inverse.numel() == n_elements, "inverse size mismatch");
    
    // Output tensor for aggregated values (one per unique key)
    auto options = supplycost.options();
    auto agg = torch::zeros({n_unique_keys}, options);
    
    // Choose kernel based on data size
    int blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (n_unique_keys <= BLOCK_SIZE) {
        // Use fused kernel with shared memory reduction
        size_t shared_mem_size = n_unique_keys * sizeof(double);
        q11_fused_aggregate_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>(
            supplycost.data_ptr<double>(),
            availqty.data_ptr<double>(),
            inverse.data_ptr<int64_t>(),
            agg.data_ptr<double>(),
            n_elements,
            n_unique_keys
        );
    } else {
        // Use simple kernel for large key spaces
        q11_simple_aggregate_kernel<<<blocks, BLOCK_SIZE>>>(
            supplycost.data_ptr<double>(),
            availqty.data_ptr<double>(),
            inverse.data_ptr<int64_t>(),
            agg.data_ptr<double>(),
            n_elements
        );
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    // Sync to ensure completion
    cudaDeviceSynchronize();
    
    return agg;
}

// Kernel to compute total sum of all values
__global__ void q11_total_sum_kernel(
    const double* __restrict__ supplycost_ptr,
    const double* __restrict__ availqty_ptr,
    double* __restrict__ total_ptr,
    int64_t n_elements
) {
    extern __shared__ double shared_sum[];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Each thread computes partial sum
    double my_sum = 0.0;
    if (idx < n_elements) {
        my_sum = supplycost_ptr[idx] * availqty_ptr[idx];
    }
    shared_sum[tid] = my_sum;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes block sum to global
    if (tid == 0) {
        atomicAdd(total_ptr, shared_sum[0]);
    }
}

double q11_total_sum_launcher(
    torch::Tensor supplycost,
    torch::Tensor availqty
) {
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64, "supplycost must be float64");
    TORCH_CHECK(availqty.scalar_type() == torch::kFloat64, "availqty must be float64");
    TORCH_CHECK(supplycost.is_cuda(), "supplycost must be on CUDA");
    TORCH_CHECK(availqty.is_cuda(), "availqty must be on CUDA");
    
    supplycost = supplycost.contiguous();
    availqty = availqty.contiguous();
    
    int64_t n_elements = supplycost.numel();
    TORCH_CHECK(availqty.numel() == n_elements, "availqty size mismatch");
    
    auto total = torch::zeros({1}, supplycost.options());
    int blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem_size = BLOCK_SIZE * sizeof(double);
    
    q11_total_sum_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>(
        supplycost.data_ptr<double>(),
        availqty.data_ptr<double>(),
        total.data_ptr<double>(),
        n_elements
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    
    return total.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q11_aggregate_launcher", &q11_aggregate_launcher, "TPC-H Q11 fused aggregation (CUDA)");
    m.def("q11_total_sum_launcher", &q11_total_sum_launcher, "TPC-H Q11 total sum (CUDA)");
}
'''

def _load_cuda():
    """Load CUDA extension at import time. Writes _CUDA_SOURCE to temp file and compiles."""
    tmpdir = tempfile.mkdtemp(prefix="q11_cuda_")
    cu_path = os.path.join(tmpdir, "_q11_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q11_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],  # H100 Hopper architecture
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    ps_partkey: torch.Tensor,
    ps_supplycost: torch.Tensor,
    ps_availqty: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized GPU compute for TPC-H Q11 aggregation.
    
    Runs on GPU (CUDA):
    - Fused kernel computes value = supplycost * availqty and aggregates by inverse index
    - Block-level shared memory reduction before atomicAdd minimizes global atomics
    - Total sum computed in separate fused kernel
    
    Runs on host (PyTorch):
    - torch.unique for key deduplication (already optimized)
    - Filter and sort operations
    
    Why faster than reference:
    - Fuses value computation (multiply) with aggregation in single kernel pass
    - Avoids intermediate value tensor allocation
    - Shared memory reduction reduces atomic contention on global memory
    - Two kernel strategies based on unique key count for optimal performance
    """
    # Get unique keys and inverse mapping (PyTorch unique is already optimized)
    unique_keys, inverse = torch.unique(ps_partkey, return_inverse=True)
    n_unique = unique_keys.shape[0]
    
    # Ensure tensors are contiguous and on CUDA
    supplycost = ps_supplycost.contiguous().cuda()
    availqty = ps_availqty.contiguous().cuda()
    inverse = inverse.contiguous().cuda()
    
    # Fused aggregation: compute value and aggregate in one kernel pass
    grouped_values = _EXT.q11_aggregate_launcher(
        supplycost, availqty, inverse, n_unique
    )
    
    # Compute total sum using fused kernel (avoids intermediate value tensor)
    total = _EXT.q11_total_sum_launcher(supplycost, availqty)
    threshold = total * threshold_fraction
    
    # HAVING filter (on host for simplicity and correctness)
    mask = grouped_values > threshold
    filtered_keys = unique_keys[mask]
    filtered_values = grouped_values[mask]
    
    # ORDER BY value DESC
    if filtered_values.numel() > 0:
        sort_idx = torch.argsort(filtered_values, descending=True)
        return filtered_keys[sort_idx], filtered_values[sort_idx]
    else:
        return filtered_keys, filtered_values


def run_query(
    tables: dict[str, cudf.DataFrame],
    nation_name: str = "GERMANY",
    threshold_fraction: float = 0.0001,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        GROUP BY ps_partkey
        HAVING sum(ps_supplycost * ps_availqty) > (
            SELECT sum(ps_supplycost * ps_availqty) * 0.0001
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
        )
        ORDER BY value DESC;
    Description: TPC-H Q11 - Identifies parts contributing significantly to the total
                 supply cost value for suppliers in a given nation.
    Params:
        nation_name (str): nation name filter literal.
        threshold_fraction (float): fractional threshold used in HAVING clause.
    """
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    # Filter nation
    filtered_nation = df_nation[df_nation["n_name"] == nation_name]

    # Join supplier <-> nation
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )

    # Join partsupp <-> (supplier <-> nation)
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")

    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)

    keys_t, values_t = _query_core(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    return cudf.DataFrame({
        "ps_partkey": keys_t,
        "value": values_t,
    })
