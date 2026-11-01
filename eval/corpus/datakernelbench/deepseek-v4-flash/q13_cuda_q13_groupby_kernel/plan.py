import os
import tempfile
import torch
import cudf
import pandas as pd
import numpy as np
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q13_histogram_kernel(
    const int64_t* __restrict__ o_custkey_ptr,
    int64_t* __restrict__ counts_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t custkey = o_custkey_ptr[idx];
        atomicAdd((unsigned long long*)&counts_ptr[custkey], 1ULL);
    }
}

__global__ void q13_gather_kernel(
    const int64_t* __restrict__ c_custkey_ptr,
    const int64_t* __restrict__ counts_ptr,
    int64_t* __restrict__ c_count_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t custkey = c_custkey_ptr[idx];
        c_count_ptr[idx] = counts_ptr[custkey];
    }
}

__global__ void q13_groupby_kernel(
    const int64_t* __restrict__ c_count_ptr,
    int64_t* __restrict__ custdist_out_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t count = c_count_ptr[idx];
        atomicAdd((unsigned long long*)&custdist_out_ptr[count], 1ULL);
    }
}

torch::Tensor q13_launcher(
    torch::Tensor c_custkey,
    torch::Tensor o_custkey_filtered,
    int64_t num_customers
) {
    c_custkey = c_custkey.contiguous().cuda();
    o_custkey_filtered = o_custkey_filtered.contiguous().cuda();
    
    int64_t n_cust = c_custkey.size(0);
    int64_t n_orders = o_custkey_filtered.size(0);
    
    TORCH_CHECK(c_custkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(o_custkey_filtered.scalar_type() == torch::kInt64);
    
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
    auto order_counts = torch::zeros({num_customers + 1}, options);
    
    // Step 1: Build histogram of order counts per customer
    int blocks = (n_orders + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q13_histogram_kernel<<<blocks, BLOCK_SIZE>>>(
        o_custkey_filtered.data_ptr<int64_t>(),
        order_counts.data_ptr<int64_t>(),
        n_orders
    );
    
    // Step 2: Gather counts for each customer (LEFT JOIN semantics)
    auto c_count = torch::zeros({n_cust}, options);
    blocks = (n_cust + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q13_gather_kernel<<<blocks, BLOCK_SIZE>>>(
        c_custkey.data_ptr<int64_t>(),
        order_counts.data_ptr<int64_t>(),
        c_count.data_ptr<int64_t>(),
        n_cust
    );
    
    // Find max count for histogram size
    int64_t max_count_val = order_counts.max().item<int64_t>();
    if (max_count_val < 0) max_count_val = 0;
    
    // Step 3: Group by c_count -> custdist
    auto custdist = torch::zeros({max_count_val + 1}, options);
    q13_groupby_kernel<<<blocks, BLOCK_SIZE>>>(
        c_count.data_ptr<int64_t>(),
        custdist.data_ptr<int64_t>(),
        n_cust
    );
    
    // Step 4: Collect non-zero results
    auto custdist_cpu = custdist.cpu();
    auto custdist_accessor = custdist_cpu.accessor<int64_t, 1>();
    std::vector<int64_t> unique_vals;
    std::vector<int64_t> dist_vals;
    
    for (int64_t i = 0; i <= max_count_val; i++) {
        if (custdist_accessor[i] > 0) {
            unique_vals.push_back(i);
            dist_vals.push_back(custdist_accessor[i]);
        }
    }
    
    // Sort by custdist DESC, c_count DESC
    std::vector<int64_t> indices(unique_vals.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
        if (dist_vals[a] != dist_vals[b]) return dist_vals[a] > dist_vals[b];
        return unique_vals[a] > unique_vals[b];
    });
    
    // Build output tensors
    int64_t n_results = unique_vals.size();
    auto c_count_out = torch::zeros({n_results}, options);
    auto custdist_out = torch::zeros({n_results}, options);
    
    for (int64_t i = 0; i < n_results; i++) {
        c_count_out[i] = unique_vals[indices[i]];
        custdist_out[i] = dist_vals[indices[i]];
    }
    
    return torch::cat({c_count_out.unsqueeze(1), custdist_out.unsqueeze(1)}, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q13_launcher", &q13_launcher, "TPC-H Q13 CUDA kernel");
}
'''

def _load_cuda():
    """Load CUDA extension with Q13 kernel."""
    tmpdir = tempfile.mkdtemp(prefix="q13_cuda_")
    cu_path = os.path.join(tmpdir, "_q13_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q13_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-arch=sm_90", "-std=c++17"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False
    )

_EXT = _load_cuda()

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
              ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'
            GROUP BY c_custkey
        ) AS c_orders (c_custkey, c_count)
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;
    
    Description: TPC-H Q13 - Customer distribution based on number of non-special-request orders.
    Uses cuDF for string filtering (NOT LIKE) and CUDA kernel for aggregation pipeline.
    
    Optimization strategy:
    1. cuDF handles string filtering (GPU-accelerated string ops)
    2. All numeric processing moved to fused CUDA kernel:
       - Histogram construction with atomicAdd (one atomic per order using unsigned long long cast)
       - Gather for LEFT JOIN semantics
       - Group by with histogram approach
    3. Sorting moved to CPU for simplicity (small result set)
    
    Params:
        tables: dict of cudf DataFrames containing 'customer' and 'orders'
        comment_pattern: SQL LIKE pattern to exclude orders
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    
    # Filter orders using cuDF string matching
    mask = ~df_orders["o_comment"].str.contains("special.*requests", regex=True)
    filtered_orders = df_orders[mask]
    
    # Extract keys as torch tensors on GPU
    c_custkey_t = torch.as_tensor(
        df_customer["c_custkey"].to_cupy().astype(np.int64),
        dtype=torch.int64
    )
    
    o_custkey_t = torch.as_tensor(
        filtered_orders["o_custkey"].to_cupy().astype(np.int64),
        dtype=torch.int64
    )
    
    # Max customer key for histogram size
    max_custkey = int(df_customer["c_custkey"].max())
    
    # Launch fused CUDA kernel
    result = _EXT.q13_launcher(c_custkey_t, o_custkey_t, max_custkey)
    
    # Convert back to cudf DataFrame
    c_count_dev = result[:, 0]
    custdist_dev = result[:, 1]
    
    c_count_cpu = c_count_dev.cpu().numpy()
    custdist_cpu = custdist_dev.cpu().numpy()
    
    return cudf.DataFrame({
        "c_count": c_count_cpu,
        "custdist": custdist_cpu,
    })
