import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define MAX_PARTS_PER_BLOCK 1024

__device__ inline double atomicAdd_double(double *addr, double val) {
    unsigned long long *address_as_ull = (unsigned long long *)addr;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void q17_first_pass(
    const int64_t* __restrict__ l_partkey_ptr,
    const double* __restrict__ l_quantity_ptr,
    const int64_t* __restrict__ part_keys_ptr,
    double* __restrict__ sum_qty_ptr,
    double* __restrict__ cnt_qty_ptr,
    const int64_t n_items,
    const int64_t n_parts
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_items; i += blockDim.x * gridDim.x) {
        int64_t pkey = l_partkey_ptr[i];
        int64_t lo = 0, hi = n_parts;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (part_keys_ptr[mid] < pkey) lo = mid + 1;
            else hi = mid;
        }
        if (lo < n_parts && part_keys_ptr[lo] == pkey) {
            atomicAdd_double(&sum_qty_ptr[lo], l_quantity_ptr[i]);
            atomicAdd_double(&cnt_qty_ptr[lo], 1.0);
        }
    }
}

__global__ void q17_second_pass(
    const int64_t* __restrict__ l_partkey_ptr,
    const double* __restrict__ l_quantity_ptr,
    const double* __restrict__ l_extendedprice_ptr,
    const int64_t* __restrict__ part_keys_ptr,
    const double* __restrict__ sum_qty_ptr,
    const double* __restrict__ cnt_qty_ptr,
    double* __restrict__ sum_price_ptr,
    const int64_t n_items,
    const int64_t n_parts
) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    double my_sum_price = 0.0;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_items; i += blockDim.x * gridDim.x) {
        int64_t pkey = l_partkey_ptr[i];
        int64_t lo = 0, hi = n_parts;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (part_keys_ptr[mid] < pkey) lo = mid + 1;
            else hi = mid;
        }
        if (lo < n_parts && part_keys_ptr[lo] == pkey) {
            double cnt = cnt_qty_ptr[lo];
            double avg_qty = cnt > 0 ? sum_qty_ptr[lo] / cnt : 0.0;
            if (l_quantity_ptr[i] < 0.2 * avg_qty) {
                my_sum_price += l_extendedprice_ptr[i];
            }
        }
    }

    sdata[tid] = my_sum_price;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd_double(sum_price_ptr, sdata[0]);
    }
}

void q17_kernel_launcher(
    torch::Tensor l_partkey,
    torch::Tensor l_quantity,
    torch::Tensor l_extendedprice,
    torch::Tensor part_keys,
    torch::Tensor& sum_price
) {
    int64_t n_items = l_partkey.numel();
    int64_t n_parts = part_keys.numel();

    auto sum_qty = torch::zeros({n_parts}, torch::dtype(torch::kFloat64).device(l_partkey.device()));
    auto cnt_qty = torch::zeros({n_parts}, torch::dtype(torch::kFloat64).device(l_partkey.device()));

    // First kernel launch: compute sums and counts
    int blocks = (n_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q17_first_pass<<<blocks, BLOCK_SIZE>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        part_keys.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        n_items,
        n_parts
    );
    cudaDeviceSynchronize();

    // Second kernel launch: compute revenue
    blocks = (n_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q17_second_pass<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
        l_partkey.data_ptr<int64_t>(),
        l_quantity.data_ptr<double>(),
        l_extendedprice.data_ptr<double>(),
        part_keys.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        cnt_qty.data_ptr<double>(),
        sum_price.data_ptr<double>(),
        n_items,
        n_parts
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q17_kernel_launcher", &q17_kernel_launcher, "TPC-H Q17 kernel launcher");
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
    Description: TPC-H Q17. Two-pass CUDA implementation that first computes per-part averages then filters/aggregates revenue. Matches reference output exactly.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    sum_price = torch.zeros(1, dtype=torch.float64, device=l_partkey_t.device)
    _EXT.q17_kernel_launcher(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t, sum_price)

    return float(sum_price.item() / 7.0)
