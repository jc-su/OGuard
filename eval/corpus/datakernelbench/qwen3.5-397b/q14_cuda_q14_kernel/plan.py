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

__global__ void q14_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ total_ptr,
    double* __restrict__ promo_ptr_out,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        bool in_range = (sd >= date_lo) && (sd < date_hi);
        
        if (in_range) {
            double price = price_ptr[idx];
            double disc = discount_ptr[idx];
            double revenue = price * (1.0 - disc);
            my_total = revenue;
            
            if (promo_ptr[idx]) {
                my_promo = revenue;
            }
        }
    }
    
    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();
    
    // Block-level reduction for total
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(total_ptr, block_total[0]);
        atomicAdd(promo_ptr_out, block_promo[0]);
    }
}

std::vector<torch::Tensor> q14_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor shipdate_days,
    torch::Tensor promo_flag,
    int32_t date_lo,
    int32_t date_hi
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32, "shipdate must be int32");
    TORCH_CHECK(promo_flag.scalar_type() == torch::kBool, "promo must be bool");
    
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
    if (!promo_flag.is_cuda()) promo_flag = promo_flag.cuda();
    
    price = price.contiguous();
    discount = discount.contiguous();
    shipdate_days = shipdate_days.contiguous();
    promo_flag = promo_flag.contiguous();
    
    int64_t n = price.numel();
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto total_out = torch::zeros({1}, options);
    auto promo_out = torch::zeros({1}, options);
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q14_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        promo_flag.data_ptr<bool>(),
        date_lo,
        date_hi,
        total_out.data_ptr<double>(),
        promo_out.data_ptr<double>(),
        n
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return {total_out, promo_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q14_launcher", &q14_launcher, "TPC-H Q14 promo revenue (CUDA fused kernel)");
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

def _launch_q14(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Launch fused CUDA kernel for Q14 promo revenue calculation.
    
    Runs on GPU: date filter + revenue projection + dual aggregation (total + promo)
    in a single kernel pass with block-level reduction and one atomicAdd per block.
    
    Returns: tensor [total_revenue, promo_revenue]
    """
    results = _EXT.q14_launcher(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return torch.stack([results[0], results[1]])

def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    Compute promo revenue percentage using fused CUDA kernel.
    
    GPU execution: Single kernel pass performs:
    1. Date range filter (int32 epoch days comparison)
    2. Revenue projection: price * (1 - discount)
    3. Conditional promo sum based on p_type_is_promo flag
    4. Block-level shared-memory reduction
    5. One atomicAdd per block to global accumulators
    
    This is memory-bound (column scan), so fusion reduces HBM traffic
    compared to separate filter/project/aggregate passes.
    
    Args:
        l_extendedprice: float64 tensor of lineitem extended prices
        l_discount: float64 tensor of lineitem discounts
        l_shipdate_days: int32 tensor of shipdate as epoch days
        p_type_is_promo: bool tensor indicating promo part type
        shipdate_lo_days: inclusive lower bound (epoch days)
        shipdate_hi_days: exclusive upper bound (epoch days)
    
    Returns:
        float64 scalar: 100 * promo_revenue / total_revenue
    """
    aggregates = _launch_q14(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    total = aggregates[0]
    promo_sum = aggregates[1]
    
    # Avoid division by zero
    if total.item() == 0.0:
        return torch.tensor(0.0, dtype=torch.float64, device=total.device)
    
    return 100.0 * promo_sum / total

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
    
    Execution strategy:
    - cuDF: Join lineitem with part table, compute promo flag from p_type string match
    - CUDA: Fused kernel performs date filter, revenue calculation, and dual aggregation
      (total + promo) in a single GPU pass with block-level reduction
    - This avoids multiple kernel launches and intermediate tensor allocations
    
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables including 'lineitem' and 'part'
        shipdate_lo (str): Inclusive lower bound on l_shipdate (YYYY-MM-DD format)
        shipdate_hi (str): Exclusive upper bound on l_shipdate (YYYY-MM-DD format)
    
    Returns:
        float: Promo revenue percentage (0-100)
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Precompute promo flag on part side before join (string ops in cuDF)
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    # Join lineitem with part on partkey
    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    # Convert to GPU tensors with correct dtypes
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(
        joined["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        joined["l_discount"].to_cupy(), dtype=torch.float64
    )
    p_type_is_promo_t = torch.as_tensor(
        joined["p_type_is_promo"].astype("bool").to_cupy()
    )

    res = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(res.item())
