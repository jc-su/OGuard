import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

// Pass 1: Filter lineitems by checking partkey against the dense O(1) part_map.
// For matched rows, accumulate sum and count of quantity per partkey.
__global__ void q17_pass1_kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const int32_t* __restrict__ part_map,
    int64_t max_partkey,
    double* __restrict__ sum_qty,
    int32_t* __restrict__ cnt_qty,
    int64_t n_lineitems
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_lineitems) return;

    int64_t pk = l_partkey[i];
    if (pk >= 0 && pk <= max_partkey) {
        int32_t p_idx = part_map[pk];
        if (p_idx != -1) {
            double qty = l_quantity[i];
            atomicAdd(&sum_qty[p_idx], qty);
            atomicAdd(&cnt_qty[p_idx], 1);
        }
    }
}

// Pass 2: Iterate over lineitems again, re-check part map, compute threshold
// based on avg(qty), and conditionally accumulate block-wise revenue.
__global__ void q17_pass2_kernel(
    const int64_t* __restrict__ l_partkey,
    const double* __restrict__ l_quantity,
    const double* __restrict__ l_extendedprice,
    const int32_t* __restrict__ part_map,
    int64_t max_partkey,
    const double* __restrict__ sum_qty,
    const int32_t* __restrict__ cnt_qty,
    double* __restrict__ out_total,
    int64_t n_lineitems
) {
    __shared__ double block_rev[BLOCK_SIZE];
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (i < n_lineitems) {
        int64_t pk = l_partkey[i];
        if (pk >= 0 && pk <= max_partkey) {
            int32_t p_idx = part_map[pk];
            if (p_idx != -1) {
                double qty = l_quantity[i];
                double s_qty = sum_qty[p_idx];
                int32_t c_qty = cnt_qty[p_idx];
                
                if (c_qty > 0) {
                    double avg_qty = s_qty / (double)c_qty;
                    if (qty < 0.2 * avg_qty) {
                        my_rev = l_extendedprice[i];
                    }
                }
            }
        }
    }
    
    block_rev[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_rev[threadIdx.x] += block_rev[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Global atomic add
    if (threadIdx.x == 0 && block_rev[0] > 0.0) {
        atomicAdd(out_total, block_rev[0]);
    }
}

double q17_launcher(
    torch::Tensor l_partkey,
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor part_map,
    int64_t max_partkey,
    int64_t num_parts
) {
    TORCH_CHECK(l_partkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64);
    TORCH_CHECK(part_map.scalar_type() == torch::kInt32);

    int64_t n = l_partkey.numel();
    if (n == 0) return 0.0;

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto sum_qty = torch::zeros({num_parts}, options);
    auto cnt_qty = torch::zeros({num_parts}, options.dtype(torch::kInt32));
    auto out_total = torch::zeros({1}, options);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q17_pass1_kernel<<<blocks, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        part_map.data_ptr<int32_t>(),
        max_partkey,
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<int32_t>(),
        n
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    q17_pass2_kernel<<<blocks, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        part_map.data_ptr<int32_t>(),
        max_partkey,
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<int32_t>(),
        out_total.data_ptr<double>(),
        n
    );
    
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out_total.item<double>() / 7.0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_launcher", &q17_launcher, "TPC-H Q17 CUDA launcher");
}
'''

def _load_cuda() -> object:
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
    Description: TPC-H Q17. Correlated subquery is resolved via grouped average across all 
        lineitems for matching parts. Fused CUDA kernels use a dense hash map for fast O(1) partkey
        filtering and accumulation over two passes to avoid writing materialized intermediate indices.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table directly in cuDF
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]
    
    if len(df_part_f) == 0:
        return 0.0
        
    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64).contiguous()
    num_parts = part_keys_t.numel()
    max_partkey = int(torch.max(part_keys_t).item())
    
    # Dense O(1) mapping array: maps part_key -> sequential index [0, num_parts - 1]
    part_map = torch.full((max_partkey + 1,), -1, dtype=torch.int32, device="cuda")
    part_map[part_keys_t] = torch.arange(num_parts, dtype=torch.int32, device="cuda")
    
    # Extract lineitem columns
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64).contiguous()
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64).contiguous()
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64).contiguous()

    # Launch fused CUDA passes
    res = _EXT.q17_launcher(
        l_partkey_t,
        l_quantity_t,
        l_extendedprice_t,
        part_map,
        max_partkey,
        num_parts
    )
    
    return float(res)
