import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void promo_fuse_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* total_ptr,
    double* promo_ptr_out,
    int64_t n_elements
) {
    __shared__ double s_total[BLOCK_SIZE];
    __shared__ double s_promo[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        bool keep = (sd >= date_lo) && (sd < date_hi);
        if (keep) {
            double revenue = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            my_total = revenue;
            if (promo_ptr[idx]) my_promo = revenue;
        }
    }
    s_total[threadIdx.x] = my_total;
    s_promo[threadIdx.x] = my_promo;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_total[threadIdx.x] += s_total[threadIdx.x + s];
            s_promo[threadIdx.x] += s_promo[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(total_ptr, s_total[0]);
        atomicAdd(promo_ptr_out, s_promo[0]);
    }
}

double promo_revenue_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor shipdate_days,
    torch::Tensor is_promo,
    int32_t date_lo,
    int32_t date_hi
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32);
    TORCH_CHECK(is_promo.scalar_type() == torch::kBool);
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
    if (!is_promo.is_cuda()) is_promo = is_promo.cuda();
    price = price.contiguous();
    discount = discount.contiguous();
    shipdate_days = shipdate_days.contiguous();
    is_promo = is_promo.contiguous();
    int64_t n = price.numel();
    auto total_out = torch::zeros({1}, price.options());
    auto promo_out = torch::zeros({1}, price.options());
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    promo_fuse_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        is_promo.data_ptr<bool>(),
        date_lo, date_hi,
        total_out.data_ptr<double>(),
        promo_out.data_ptr<double>(),
        n
    );
    double total = total_out.item<double>();
    double promo_sum = promo_out.item<double>();
    return 100.0 * promo_sum / total;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("promo_revenue_launcher", &promo_revenue_launcher, "CUDA fused filter+agg for promo revenue ratio");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="qXX_cuda_")
    cu_path = os.path.join(tmpdir, "_promo_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="promo_cuda_ext",
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
    revival = _EXT.promo_revenue_launcher(
        l_extendedprice, l_discount, l_shipdate_days, p_type_is_promo,
        shipdate_lo_days, shipdate_hi_days
    )
    return torch.tensor(revival, dtype=torch.float64, device=l_extendedprice.device)

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
