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
#define WARP_SIZE 32

__device__ double warp_reduce(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void q17_kernel(
    const int64_t* __restrict__ l_partkey_ptr,
    const double* __restrict__ l_quantity_ptr,
    const double* __restrict__ l_extendedprice_ptr,
    const int64_t* __restrict__ part_keys_ptr,
    const double* __restrict__ avg_qty_ptr,
    const double* __restrict__ threshold_ptr,
    double* __restrict__ out_ptr,
    int64_t n_lineitems,
    int64_t n_parts
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (idx < n_lineitems) {
        int64_t pk = l_partkey_ptr[idx];
        double qty = l_quantity_ptr[idx];
        double price = l_extendedprice_ptr[idx];
        
        // Binary search for partkey in sorted part_keys
        int64_t lo = 0, hi = n_parts;
        int64_t found_idx = -1;
        
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            int64_t mid_val = part_keys_ptr[mid];
            if (mid_val == pk) {
                found_idx = mid;
                break;
            } else if (mid_val < pk) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        
        if (found_idx >= 0) {
            double avg = avg_qty_ptr[found_idx];
            double thresh = 0.2 * avg;
            if (qty < thresh) {
                my_rev = price;
            }
        }
    }
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }
    
    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}

// First pass: compute per-partkey sum and count
__global__ void compute_part_stats_kernel(
    const int64_t* __restrict__ l_partkey_ptr,
    const double* __restrict__ l_quantity_ptr,
    const int64_t* __restrict__ part_keys_ptr,
    double* __restrict__ sum_qty_ptr,
    double* __restrict__ cnt_qty_ptr,
    int64_t n_lineitems,
    int64_t n_parts
) {
    __shared__ double block_sum[BLOCK_SIZE];
    __shared__ double block_cnt[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_sum = 0.0;
    double my_cnt = 0.0;
    
    if (idx < n_lineitems) {
        int64_t pk = l_partkey_ptr[idx];
        double qty = l_quantity_ptr[idx];
        
        // Binary search for partkey
        int64_t lo = 0, hi = n_parts;
        int64_t found_idx = -1;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            int64_t mid_val = part_keys_ptr[mid];
            if (mid_val == pk) {
                found_idx = mid;
                break;
            } else if (mid_val < pk) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        
        if (found_idx >= 0) {
            my_sum = qty;
            my_cnt = 1.0;
        }
    }
    
    // Use atomicAdd per thread for scattered indices (each thread may map to different partkey)
    // This is necessary because lineitems are not sorted by partkey
    if (idx < n_lineitems) {
        int64_t pk = l_partkey_ptr[idx];
        int64_t lo = 0, hi = n_parts;
        int64_t found_idx = -1;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            int64_t mid_val = part_keys_ptr[mid];
            if (mid_val == pk) {
                found_idx = mid;
                break;
            } else if (mid_val < pk) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (found_idx >= 0) {
            atomicAdd(&sum_qty_ptr[found_idx], l_quantity_ptr[idx]);
            atomicAdd(&cnt_qty_ptr[found_idx], 1.0);
        }
    }
}

double q17_launcher(
    torch::Tensor l_partkey,
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor part_keys
) {
    TORCH_CHECK(l_partkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64);
    TORCH_CHECK(part_keys.scalar_type() == torch::kInt64);
    
    if (!l_partkey.is_cuda()) l_partkey = l_partkey.cuda();
    if (!l_quantity.is_cuda()) l_quantity = l_quantity.cuda();
    if (!l_extendedprice.is_cuda()) l_extendedprice = l_extendedprice.cuda();
    if (!part_keys.is_cuda()) part_keys = part_keys.cuda();
    
    l_partkey = l_partkey.contiguous();
    l_quantity = l_quantity.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    part_keys = part_keys.contiguous();
    
    int64_t n_lineitems = l_partkey.numel();
    int64_t n_parts = part_keys.numel();
    
    // Allocate per-partkey statistics
    auto sum_qty = torch::zeros({n_parts}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    auto cnt_qty = torch::zeros({n_parts}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    
    // First pass: compute sum and count per partkey using atomicAdd
    int blocks_part = (n_lineitems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_part_stats_kernel<<<blocks_part, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        part_keys.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        n_lineitems,
        n_parts
    );
    
    // Compute average quantity per partkey (on host side via tensor ops)
    auto avg_qty = sum_qty / cnt_qty;
    avg_qty = torch::where(cnt_qty > 0, avg_qty, torch::zeros_like(avg_qty));
    auto threshold = 0.2 * avg_qty;
    
    // Second pass: filter and sum extendedprice
    auto out = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    
    int blocks_agg = (n_lineitems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q17_kernel<<<blocks_agg, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        part_keys.data_ptr<int64_t>(),
        avg_qty.data_ptr<double>(),
        threshold.data_ptr<double>(),
        out.data_ptr<double>(),
        n_lineitems,
        n_parts
    );
    
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_launcher", &q17_launcher, "TPC-H Q17 avg_yearly (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q17_cuda_")
    cu_path = os.path.join(tmpdir, "_q17_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q17_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                            FROM lineitem
                            WHERE l_partkey = p_partkey);
    Description: TPC-H Q17. Uses cuDF for string filtering on part table (brand/container),
        then converts to GPU tensors and performs two CUDA kernel passes: first computes
        per-partkey lineitem quantity sum/count using atomicAdd; then filters lineitems
        where quantity < 0.2 * avg(quantity) for that partkey, summing extendedprice.
        This avoids scatter_add operations and multiple temporary tensors, performing
        the correlated subquery logic directly on GPU with fused filter+aggregate.
    Params:
        tables: dict of cudf DataFrames containing 'lineitem' and 'part'
        brand_val: brand filter for p_brand (default 'Brand#23')
        container_val: container filter for p_container (default 'MED BOX')
    Returns:
        float: avg_yearly = sum(l_extendedprice for qualifying lineitems) / 7.0
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]
    
    # Filter part table (string filtering in cuDF)
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]
    
    # Convert to GPU tensors
    part_keys = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    
    total = _EXT.q17_launcher(l_partkey, l_quantity, l_extendedprice, part_keys)
    return float(total) / 7.0
