import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 1024

__global__ void q11_agg_kernel(
    const int64_t* __restrict__ ps_partkey,
    const double* __restrict__ ps_supplycost,
    const double* __restrict__ ps_availqty,
    int64_t* __restrict__ unique_keys_out,
    double* __restrict__ grouped_values_out,
    int64_t* __restrict__ key_count_out,
    double* __restrict__ total_rev_ptr,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    int64_t my_key = 0;
    
    if (idx < n) {
        my_key = ps_partkey[idx];
        my_rev = ps_supplycost[idx] * ps_availqty[idx];
    }

    // Block reduce for total
    __shared__ double block_totals[BLOCK_SIZE];
    block_totals[threadIdx.x] = my_rev;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_totals[threadIdx.x] += block_totals[threadIdx.x + s];
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
        atomicAdd(total_rev_ptr, block_totals[0]);

    // Use shared memory hash table for grouping
    __shared__ double bucket_vals[2048];
    __shared__ int64_t bucket_keys[2048];
    __shared__ int bucket_used[2048];
    
    if (threadIdx.x < 2048) {
        bucket_vals[threadIdx.x] = 0.0;
        bucket_keys[threadIdx.x] = -1;
        bucket_used[threadIdx.x] = 0;
    }
    __syncthreads();
    
    if (idx < n) {
        unsigned int hash = (unsigned int)(my_key % 2048);
        
        while (atomicCAS(&bucket_used[hash], 0, 1) != 0) {
            if (bucket_keys[hash] == my_key) {
                atomicAdd(&bucket_vals[hash], my_rev);
                goto done;
            }
            hash = (hash + 1) % 2048;
        }
        bucket_keys[hash] = my_key;
        bucket_vals[hash] = my_rev;
    }
    done:;
    
    __syncthreads();
    
    if (threadIdx.x < 2048 && bucket_used[threadIdx.x]) {
        int64_t pos = atomicAdd((unsigned long long*)key_count_out, 1ULL);
        unique_keys_out[pos] = bucket_keys[threadIdx.x];
        grouped_values_out[pos] = bucket_vals[threadIdx.x];
    }
}

std::vector<torch::Tensor> q11_launcher(
    torch::Tensor ps_partkey,
    torch::Tensor ps_supplycost,
    torch::Tensor ps_availqty,
    double threshold_fraction
) {
    TORCH_CHECK(ps_partkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(ps_supplycost.scalar_type() == torch::kFloat64);
    TORCH_CHECK(ps_availqty.scalar_type() == torch::kFloat64);

    if (!ps_partkey.is_cuda()) ps_partkey = ps_partkey.cuda();
    if (!ps_supplycost.is_cuda()) ps_supplycost = ps_supplycost.cuda();
    if (!ps_availqty.is_cuda()) ps_availqty = ps_availqty.cuda();

    ps_partkey = ps_partkey.contiguous();
    ps_supplycost = ps_supplycost.contiguous();
    ps_availqty = ps_availqty.contiguous();

    int64_t n = ps_partkey.numel();

    // Allocate output buffers
    auto unique_keys = torch::empty({n}, ps_partkey.options());
    auto grouped_values = torch::zeros({n}, ps_supplycost.options());
    auto key_count = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto total_rev = torch::zeros({1}, ps_supplycost.options());

    // Launch aggregation kernel
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q11_agg_kernel<<<blocks, BLOCK_SIZE>>>(
        ps_partkey.data_ptr<int64_t>(),
        ps_supplycost.data_ptr<double>(),
        ps_availqty.data_ptr<double>(),
        unique_keys.data_ptr<int64_t>(),
        grouped_values.data_ptr<double>(),
        key_count.data_ptr<int64_t>(),
        total_rev.data_ptr<double>(),
        n
    );

    // Get actual count and threshold
    int64_t actual_count = key_count.item<int64_t>();
    double total = total_rev.item<double>();
    double threshold = total * threshold_fraction;

    // Trim to actual count
    auto trimmed_keys = unique_keys.slice(0, 0, actual_count);
    auto trimmed_vals = grouped_values.slice(0, 0, actual_count);

    // Filter by threshold
    auto mask = trimmed_vals > threshold;
    auto filtered_keys = trimmed_keys.masked_select(mask);
    auto filtered_vals = trimmed_vals.masked_select(mask);

    // Sort by value descending using PyTorch's sort on GPU
    auto sort_idx = std::get<1>(filtered_vals.sort(0, true));
    auto result_keys = filtered_keys.index_select(0, sort_idx);
    auto result_vals = filtered_vals.index_select(0, sort_idx);

    return {result_keys.to(torch::kCPU), result_vals.to(torch::kCPU)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q11_launcher", &q11_launcher, "TPC-H Q11 aggregation (CUDA)");
}
'''

def _load_cuda():
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
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

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
    Description: TPC-H Q11 with fused CUDA kernel for aggregation.
                 Joins done in cuDF (fast for small dimension tables).
                 All aggregation, filtering to a single compiled CUDA extension
                 on the GPU, reducing kernel launch overhead and data movement.
                 Uses shared memory hash table for grouping by ps_partkey.
    Params:
        tables (dict): Input tables as cudf DataFrames.
        nation_name (str): Nation filter. Default 'GERMANY'.
        threshold_fraction (float): Fraction for HAVING threshold. Default 0.0001.
    Returns:
        cudf.DataFrame: Result with columns ps_partkey and value.
    """
    # cuDF joins (small tables, fast)
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]

    filtered_nation = df_nation[df_nation["n_name"] == nation_name]
    sup_nat = df_supplier.merge(
        filtered_nation, left_on="s_nationkey", right_on="n_nationkey"
    )
    joined = df_partsupp.merge(sup_nat, left_on="ps_suppkey", right_on="s_suppkey")

    # Convert to tensors
    ps_partkey_t = torch.as_tensor(joined["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    ps_availqty_t = torch.as_tensor(joined["ps_availqty"].to_cupy(), dtype=torch.float64)

    # Launch fused CUDA kernel
    keys_t, values_t = _EXT.q11_launcher(
        ps_partkey_t, ps_supplycost_t, ps_availqty_t, threshold_fraction
    )

    return cudf.DataFrame({
        "ps_partkey": keys_t,
        "value": values_t,
    })
