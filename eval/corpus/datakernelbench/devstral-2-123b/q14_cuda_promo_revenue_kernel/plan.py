import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_fp16.h> // Not needed here, but common include pattern

#define BLOCK_SIZE 1024

__global__ void promo_revenue_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double total_rev_out[],
    double promo_rev_out[],
    int64_t n_elements
) {
    __shared__ double block_total_rev[BLOCK_SIZE];
    __shared__ double block_promo_rev[BLOCK_SIZE];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        bool keep = (sd >= date_lo) && (sd < date_hi);
        if (keep) {
            double price = price_ptr[idx];
            double disc = discount_ptr[idx];
            double rev = price * (1.0 - disc);
            my_total = rev;
            if (promo_ptr[idx]) my_promo = rev;
        }
    }
    block_total_rev[threadIdx.x] = my_total;
    block_promo_rev[threadIdx.x] = my_promo;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_total_rev[threadIdx.x] += block_total_rev[threadIdx.x + s];
            block_promo_rev[threadIdx.x] += block_promo_rev[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd((double*)total_rev_out, block_total_rev[0]);
        atomicAdd((double*)promo_rev_out, block_promo_rev[0]);
    }
}

torch::Tensor promo_revenue_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor shipdate_days,
    torch::Tensor promo_flags,
    int32_t date_lo,
    int32_t date_hi
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
    if (!promo_flags.is_cuda()) promo_flags = promo_flags.cuda();
    price = price.contiguous();
    discount = discount.contiguous();
    shipdate_days = shipdate_days.contiguous();
    promo_flags = promo_flags.contiguous();
    int64_t n = price.numel();
    auto total_rev = torch::zeros({1}, price.options());
    auto promo_rev = torch::zeros({1}, price.options());
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    promo_revenue_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        promo_flags.data_ptr<bool>(),
        date_lo, date_hi,
        total_rev.data_ptr<double>(),
        promo_rev.data_ptr<double>(),
        n
    );
    return torch::stack({total_rev, promo_rev}, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("promo_revenue", &promo_revenue_launcher, "Fused CUDA launcher for promo revenue ratio");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="promo_rev_cuda_x")
    cu_path = os.path.join(tmpdir, "_promo_rev_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    ext = load(
        name="promo_rev_cuda_ext_x",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )
    return ext

_EXT = _load_cuda()

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    Description: Computes the percentage of revenue from promotional parts shipped between two dates.
                 Pipeline: 1) cuDF join and promo flag, 2) GPU tensorize, 3) single fused CUDA kernel (filter + project + reduce) with block shared-memory reductions.
                 Expected performance: memory-bound improvement by fusing filter, project, and aggregation into one coalesced pass and avoiding temporary GPU buffers.
    Params:
        tables (dict[str, cudf.DataFrame]): in-memory cuDF datasets: keys 'lineitem', 'part'.
        shipdate_lo (str): inclusive lower bound on l_shipdate in 'YYYY-MM-DD' (default '1995-09-01').
        shipdate_hi (str): exclusive upper bound on l_shipdate in 'YYYY-MM-DD' (default '1995-10-01').
    Returns:
        float: the ratio multiplied by 100.0.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })
    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    f64_options = {"dtype": torch.float64}

    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), **f64_options)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), **f64_options)
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    total_promo_t = _EXT.promo_revenue(
        l_extendedprice_t, l_discount_t, l_shipdate_days_t, p_type_is_promo_t,
        shipdate_lo_days, shipdate_hi_days
    )
    total_rev, promo_rev = total_promo_t.float().cpu()
    return float(100.0 * promo_rev.item() / total_rev.item())
