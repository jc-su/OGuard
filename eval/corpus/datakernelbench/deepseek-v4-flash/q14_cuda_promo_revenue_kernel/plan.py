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

__global__ void promo_revenue_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_flag_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ total_ptr,
    double* __restrict__ promo_ptr,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        bool keep = (sd >= date_lo) && (sd < date_hi);
        if (keep) {
            double rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            my_total = rev;
            if (promo_flag_ptr[idx]) {
                my_promo = rev;
            }
        }
    }
    
    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(total_ptr, block_total[0]);
        atomicAdd(promo_ptr, block_promo[0]);
    }
}

double promo_revenue_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor shipdate_days,
    torch::Tensor promo_flag,
    int32_t date_lo,
    int32_t date_hi
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(promo_flag.scalar_type() == torch::kBool);
    
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
    if (!promo_flag.is_cuda()) promo_flag = promo_flag.cuda();
    
    price = price.contiguous();
    discount = discount.contiguous();
    shipdate_days = shipdate_days.contiguous();
    promo_flag = promo_flag.contiguous();
    
    int64_t n = price.numel();
    auto total = torch::zeros({1}, price.options());
    auto promo = torch::zeros({1}, price.options());
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    promo_revenue_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        promo_flag.data_ptr<bool>(),
        date_lo, date_hi,
        total.data_ptr<double>(),
        promo.data_ptr<double>(),
        n
    );
    
    double total_val = total.item<double>();
    double promo_val = promo.item<double>();
    
    if (total_val == 0.0) return 0.0;
    return 100.0 * promo_val / total_val;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("promo_revenue_launcher", &promo_revenue_launcher, "TPC-H Promo Revenue (CUDA fused)");
}
'''

def _load_cuda():
    """Compile and load the CUDA extension once at module import time."""
    tmpdir = tempfile.mkdtemp(prefix="q14_cuda_")
    cu_path = os.path.join(tmpdir, "_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q14_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
    Optimized with a fused CUDA kernel that performs filter, projection, and aggregation in a single pass.
    The cuDF join and string matching run on GPU, then tensors are moved to CUDA for the fused kernel.
    
    Params:
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Precompute promo flag on part side using cuDF GPU strings
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    # Extract columns and convert to GPU tensors
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    # Launch fused CUDA kernel
    result = _EXT.promo_revenue_launcher(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(result)
