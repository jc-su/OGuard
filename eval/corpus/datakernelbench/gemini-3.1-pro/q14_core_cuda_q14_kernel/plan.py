import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <vector>

#define BLOCK_SIZE 1024

__global__ void q14_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ out_total,
    double* __restrict__ out_promo,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;

    // Conditionally read price, discount, and promo flags only if the shipdate 
    // filter passes. This saves a massive amount of memory bandwidth because 
    // selectivity is very low (~1.2% for a one-month window).
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        if (sd >= date_lo && sd < date_hi) {
            double price = price_ptr[idx];
            double discount = discount_ptr[idx];
            bool promo = promo_ptr[idx];
            
            double rev = price * (1.0 - discount);
            my_total += rev;
            if (promo) {
                my_promo += rev;
            }
        }
    }

    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();

    // Block-level reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Single global atomic per block
    if (threadIdx.x == 0) {
        atomicAdd(out_total, block_total[0]);
        atomicAdd(out_promo, block_promo[0]);
    }
}

std::vector<torch::Tensor> q14_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor shipdate,
    torch::Tensor promo,
    int32_t date_lo,
    int32_t date_hi
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(shipdate.scalar_type() == torch::kInt32);
    TORCH_CHECK(promo.scalar_type() == torch::kBool);

    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!shipdate.is_cuda()) shipdate = shipdate.cuda();
    if (!promo.is_cuda()) promo = promo.cuda();
    
    price = price.contiguous();
    discount = discount.contiguous();
    shipdate = shipdate.contiguous();
    promo = promo.contiguous();
    
    int64_t n = price.numel();
    auto options = price.options();
    
    // Create scalar 0-d tensors for outputs
    auto out_total = torch::zeros({}, options);
    auto out_promo = torch::zeros({}, options);
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 0) {
        q14_kernel<<<blocks, BLOCK_SIZE>>>(
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            shipdate.data_ptr<int32_t>(),
            promo.data_ptr<bool>(),
            date_lo,
            date_hi,
            out_total.data_ptr<double>(),
            out_promo.data_ptr<double>(),
            n
        );
    }
    
    return {out_promo, out_total};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q14_launcher", &q14_launcher, "TPC-H Q14 revenue (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q14_cuda_")
    cu_path = os.path.join(tmpdir, "_q14_kernels.cu")
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


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Fused CUDA kernel execution for TPC-H Q14.
    Replaces eager masking and sum aggregations with a single pass that conditionally
    reads the projection columns to heavily reduce memory bandwidth.
    """
    promo_sum, total = _EXT.q14_launcher(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days
    )
    return 100.0 * promo_sum / total


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
    Params:
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Precompute promo flag on part side before join
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    res = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(res.item())
