import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void promo_revenue_kernel(
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const bool* __restrict__ p_type_is_promo,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ total_revenue,
    double* __restrict__ promo_revenue,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    
    if (idx < n_elements) {
        int32_t sd = l_shipdate_days[idx];
        // Filter: shipdate_lo <= sd < shipdate_hi
        if (sd >= shipdate_lo_days && sd < shipdate_hi_days) {
            double price = l_extendedprice[idx];
            double disc = l_discount[idx];
            double revenue = price * (1.0 - disc);
            my_total = revenue;
            if (p_type_is_promo[idx]) {
                my_promo = revenue;
            }
        }
    }
    
    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total[threadIdx.x] += block_total[threadIdx.x + s];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block atomically adds to global sums
    if (threadIdx.x == 0) {
        atomicAdd(total_revenue, block_total[0]);
        atomicAdd(promo_revenue, block_promo[0]);
    }
}

std::vector<double> promo_revenue_launcher(
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    torch::Tensor p_type_is_promo,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days
) {
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_shipdate_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(p_type_is_promo.scalar_type() == torch::kBool);
    
    if (!l_extendedprice.is_cuda()) l_extendedprice = l_extendedprice.cuda();
    if (!l_discount.is_cuda()) l_discount = l_discount.cuda();
    if (!l_shipdate_days.is_cuda()) l_shipdate_days = l_shipdate_days.cuda();
    if (!p_type_is_promo.is_cuda()) p_type_is_promo = p_type_is_promo.cuda();
    
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();
    p_type_is_promo = p_type_is_promo.contiguous();
    
    int64_t n = l_extendedprice.numel();
    auto total_out = torch::zeros({1}, l_extendedprice.options());
    auto promo_out = torch::zeros({1}, l_extendedprice.options());
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    promo_revenue_kernel<<<blocks, BLOCK_SIZE>>>(
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate_days.data_ptr<int32_t>(),
        p_type_is_promo.data_ptr<bool>(),
        shipdate_lo_days,
        shipdate_hi_days,
        total_out.data_ptr<double>(),
        promo_out.data_ptr<double>(),
        n
    );
    
    // Handle empty result (avoid division by zero)
    double total = total_out.item<double>();
    double promo = promo_out.item<double>();
    if (total == 0.0) {
        return {0.0, 0.0};
    }
    return {promo, total};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("promo_revenue_launcher", &promo_revenue_launcher, 
          "TPC-H promotional revenue calculation (CUDA)");
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
    CUDA hot path: fused filter + project + aggregate.
    Computes promotional revenue percentage in a single kernel launch.
    """
    result = _EXT.promo_revenue_launcher(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    promo, total = result[0], result[1]
    return torch.tensor(100.0 * promo / total, dtype=torch.float64)


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
    This implementation uses cuDF for the join (string prefix matching) and a fused CUDA kernel
    for the hot path: filter by shipdate, compute revenue, differentiate promo vs regular,
    and aggregate in a single kernel with block-level reduction.
    
    Params:
        tables (dict): Dictionary containing 'lineitem' and 'part' DataFrames.
        shipdate_lo (str): Inclusive lower bound on l_shipdate (YYYY-MM-DD).
        shipdate_hi (str): Exclusive upper bound on l_shipdate (YYYY-MM-DD).
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # cuDF join: lineitem + part on partkey, precompute promo flag
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    # Convert to GPU tensors - float64 for monetary values, int32 for date epoch days
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    # Launch fused CUDA kernel for filter + aggregate
    res = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(res.item())
