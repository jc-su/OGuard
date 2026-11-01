import os
import tempfile
from typing import Any

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q14_aggregate_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const bool* __restrict__ promo_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ sums_ptr,
    int64_t n_elements
) {
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double my_total = 0.0;
    double my_promo = 0.0;

    if (idx < n_elements) {
        const int32_t sd = shipdate_ptr[idx];
        if ((sd >= date_lo) && (sd < date_hi)) {
            const double revenue = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            my_total = revenue;
            if (promo_ptr[idx]) {
                my_promo = revenue;
            }
        }
    }

    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_total[threadIdx.x] += block_total[threadIdx.x + stride];
            block_promo[threadIdx.x] += block_promo[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sums_ptr + 0, block_total[0]);
        atomicAdd(sums_ptr + 1, block_promo[0]);
    }
}

__global__ void q14_finalize_kernel(
    const double* __restrict__ sums_ptr,
    double* __restrict__ out_ptr
) {
    out_ptr[0] = 100.0 * sums_ptr[1] / sums_ptr[0];
}

torch::Tensor q14_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor shipdate_days,
    torch::Tensor promo,
    int32_t date_lo,
    int32_t date_hi
) {
    TORCH_CHECK(price.is_cuda(), "price must be a CUDA tensor");
    TORCH_CHECK(discount.is_cuda(), "discount must be a CUDA tensor");
    TORCH_CHECK(shipdate_days.is_cuda(), "shipdate_days must be a CUDA tensor");
    TORCH_CHECK(promo.is_cuda(), "promo must be a CUDA tensor");

    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32, "shipdate_days must be int32 epoch days");
    TORCH_CHECK(promo.scalar_type() == torch::kBool, "promo must be bool");

    TORCH_CHECK(price.dim() == 1, "price must be 1D");
    TORCH_CHECK(discount.dim() == 1, "discount must be 1D");
    TORCH_CHECK(shipdate_days.dim() == 1, "shipdate_days must be 1D");
    TORCH_CHECK(promo.dim() == 1, "promo must be 1D");

    TORCH_CHECK(discount.numel() == price.numel(), "discount length mismatch");
    TORCH_CHECK(shipdate_days.numel() == price.numel(), "shipdate_days length mismatch");
    TORCH_CHECK(promo.numel() == price.numel(), "promo length mismatch");

    price = price.contiguous();
    discount = discount.contiguous();
    shipdate_days = shipdate_days.contiguous();
    promo = promo.contiguous();

    const int64_t n = price.numel();
    auto sums = torch::zeros({2}, price.options());
    auto out = torch::empty({1}, price.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n > 0) {
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q14_aggregate_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            shipdate_days.data_ptr<int32_t>(),
            promo.data_ptr<bool>(),
            date_lo,
            date_hi,
            sums.data_ptr<double>(),
            n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    q14_finalize_kernel<<<1, 1, 0, stream>>>(
        sums.data_ptr<double>(),
        out.data_ptr<double>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out.squeeze();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q14_launcher", &q14_launcher, "TPC-H Q14 promo revenue fused CUDA core");
}
'''


def _load_cuda() -> Any:
    """
    Compile the CUDA extension from an in-module source string.

    The device code runs the numeric hot path on the GPU as one fused scan and
    reduction.  Writing the source to a temporary on-disk .cu file lets
    torch.utils.cpp_extension.load invoke nvcc while keeping the submission as a
    single Python module.
    """
    tmpdir = tempfile.mkdtemp(prefix="q14_cuda_")
    cu_path = os.path.join(tmpdir, "_q14_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q14_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
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
    GPU CUDA hot path for TPC-H Q14 after cuDF has performed the join.

    The reference eager PyTorch core materializes a date mask, a revenue vector,
    filtered vectors, and then runs separate reductions.  This implementation
    launches a hand-written CUDA kernel that scans aligned 1D tensors once,
    applies the date predicate in registers, conditionally computes revenue, and
    reduces total/promo revenue with one global atomic per block.  A tiny GPU
    finalize kernel computes the returned scalar percentage, avoiding large
    temporaries and extra HBM passes for this memory-bound column scan.
    """
    return _EXT.q14_launcher(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        int(shipdate_lo_days),
        int(shipdate_hi_days),
    )


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
