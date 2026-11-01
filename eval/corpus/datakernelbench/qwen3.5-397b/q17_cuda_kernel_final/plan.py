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

__device__ __forceinline__ int64_t binary_search(const int64_t* arr, int64_t n, int64_t key) {
    int64_t low = 0;
    int64_t high = n;
    while (low < high) {
        int64_t mid = low + ((high - low) >> 1);
        if (arr[mid] < key) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low < n && arr[low] == key) {
        return low;
    }
    return -1;
}

__global__ void kernel_agg(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const int64_t* __restrict__ part_keys,
    double* __restrict__ sum_qty,
    double* __restrict__ count_qty,
    int64_t n_lineitems,
    int64_t n_parts
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lineitems) return;

    int64_t pkey = l_partkey[idx];
    int64_t pidx = binary_search(part_keys, n_parts, pkey);
    if (pidx != -1) {
        double qty = l_quantity[idx];
        atomicAdd(&sum_qty[pidx], qty);
        atomicAdd(&count_qty[pidx], 1.0);
    }
}

__global__ void kernel_final(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const int64_t* __restrict__ part_keys,
    const double* __restrict__ sum_qty,
    const double* __restrict__ count_qty,
    double* __restrict__ revenue,
    int64_t n_lineitems,
    int64_t n_parts
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_rev = 0.0;
    if (idx < n_lineitems) {
        int64_t pkey = l_partkey[idx];
        int64_t pidx = binary_search(part_keys, n_parts, pkey);
        if (pidx != -1) {
            double s = sum_qty[pidx];
            double c = count_qty[pidx];
            double avg = (c > 0.0) ? (s / c) : 0.0;
            double thresh = 0.2 * avg;
            if (l_quantity[idx] < thresh) {
                my_rev = l_extendedprice[idx];
            }
        }
    }
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(revenue, block_revenue[0]);
    }
}

double q17_launcher(
    torch::Tensor l_partkey,
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor part_keys
) {
    TORCH_CHECK(l_partkey.is_cuda(), "l_partkey must be CUDA");
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be CUDA");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be CUDA");
    TORCH_CHECK(part_keys.is_cuda(), "part_keys must be CUDA");
    
    TORCH_CHECK(l_partkey.scalar_type() == torch::kInt64, "l_partkey must be int64");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(part_keys.scalar_type() == torch::kInt64, "part_keys must be int64");
    
    l_partkey = l_partkey.contiguous();
    l_quantity = l_quantity.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    part_keys = part_keys.contiguous();
    
    int64_t n_lineitems = l_extendedprice.numel();
    int64_t n_parts = part_keys.numel();
    
    if (n_lineitems == 0 || n_parts == 0) {
        return 0.0;
    }
    
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto sum_qty = torch::zeros({n_parts}, options);
    auto count_qty = torch::zeros({n_parts}, options);
    auto revenue = torch::zeros({1}, options);
    
    int blocks = (n_lineitems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel_agg<<<blocks, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        part_keys.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        count_qty.data_ptr<double>(),
        n_lineitems,
        n_parts
    );
    
    kernel_final<<<blocks, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        part_keys.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        count_qty.data_ptr<double>(),
        revenue.data_ptr<double>(),
        n_lineitems,
        n_parts
    );
    
    cudaDeviceSynchronize();
    
    return revenue.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_launcher", &q17_launcher, "TPC-H Q17 CUDA Launcher");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q17_cuda_")
    cu_path = os.path.join(tmpdir, "_kernels.cu")
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
          AND l_quantity < (SELECT 0.2 * avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey);
    Description: TPC-H Q17. Filters part table on CPU (cuDF), then uses two fused CUDA kernels 
        to scan lineitems. Kernel 1 computes per-part sum/count of quantity. Kernel 2 computes 
        revenue for rows where quantity < 0.2 * avg_qty. Avoids large intermediate tensors by 
        performing binary search for part membership directly in CUDA kernels.
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables (lineitem, part).
        brand_val (str): Brand filter for part table.
        container_val (str): Container filter for part table.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter part table on CPU
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]
    
    if df_part_f.empty:
        return 0.0

    # Prepare part keys (sorted for binary search)
    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    part_keys_t, _ = torch.sort(part_keys_t)
    part_keys_t = part_keys_t.contiguous().cuda()

    # Prepare lineitem columns
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64).cuda()
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64).cuda()
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64).cuda()

    # Launch CUDA kernels
    res = _EXT.q17_launcher(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    
    return float(res) / 7.0
