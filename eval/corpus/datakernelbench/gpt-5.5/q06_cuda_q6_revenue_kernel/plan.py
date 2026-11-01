import os
import tempfile
from typing import Any

import cudf
import pandas as pd
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q6_revenue_kernel(
    const int32_t* __restrict__ shipdate_days_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ extendedprice_ptr,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double discount_lo,
    double discount_hi,
    double quantity_hi,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double my_revenue = 0.0;

    if (idx < n_elements) {
        const int32_t shipdate_days = shipdate_days_ptr[idx];

        if ((shipdate_days >= shipdate_lo_days) && (shipdate_days < shipdate_hi_days)) {
            const double discount = discount_ptr[idx];

            if ((discount >= discount_lo) && (discount <= discount_hi)) {
                const double quantity = quantity_ptr[idx];

                if (quantity < quantity_hi) {
                    const double extendedprice = extendedprice_ptr[idx];
                    my_revenue = extendedprice * discount;
                }
            }
        }
    }

    block_revenue[threadIdx.x] = my_revenue;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}

double q6_revenue_launcher(
    torch::Tensor shipdate_days,
    torch::Tensor discount,
    torch::Tensor quantity,
    torch::Tensor extendedprice,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double discount_lo,
    double discount_hi,
    double quantity_hi
) {
    TORCH_CHECK(shipdate_days.is_cuda(), "shipdate_days must be a CUDA tensor");
    TORCH_CHECK(discount.is_cuda(), "discount must be a CUDA tensor");
    TORCH_CHECK(quantity.is_cuda(), "quantity must be a CUDA tensor");
    TORCH_CHECK(extendedprice.is_cuda(), "extendedprice must be a CUDA tensor");

    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32, "shipdate_days must be int32 epoch days");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(extendedprice.scalar_type() == torch::kFloat64, "extendedprice must be float64");

    const int64_t n_elements = shipdate_days.numel();
    TORCH_CHECK(discount.numel() == n_elements, "discount length mismatch");
    TORCH_CHECK(quantity.numel() == n_elements, "quantity length mismatch");
    TORCH_CHECK(extendedprice.numel() == n_elements, "extendedprice length mismatch");

    if (n_elements == 0) {
        return 0.0;
    }

    shipdate_days = shipdate_days.contiguous();
    discount = discount.contiguous();
    quantity = quantity.contiguous();
    extendedprice = extendedprice.contiguous();

    auto out = torch::zeros({1}, extendedprice.options());

    const int blocks = static_cast<int>((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    q6_revenue_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        shipdate_days.data_ptr<int32_t>(),
        discount.data_ptr<double>(),
        quantity.data_ptr<double>(),
        extendedprice.data_ptr<double>(),
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
        out.data_ptr<double>(),
        n_elements
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q6_revenue_launcher", &q6_revenue_launcher, "TPC-H Q6 fused filter/project/reduce CUDA launcher");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension from an in-module .cu string at import time."""
    tmpdir = tempfile.mkdtemp(prefix="q6_cuda_")
    cu_path = os.path.join(tmpdir, "q6_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)

    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q6_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _launch_q6_revenue(
    l_shipdate_days: torch.Tensor,
    l_discount: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    discount_lo: float,
    discount_hi: float,
    quantity_hi: float,
) -> torch.Tensor:
    """
    Runs on GPU through the compiled CUDA extension.

    The kernel fuses date/discount/quantity filtering, revenue projection, and
    block-level reduction into one memory-bound pass, avoiding PyTorch mask
    materialization and multiple temporary tensors.
    """
    revenue = _EXT.q6_revenue_launcher(
        l_shipdate_days,
        l_discount,
        l_quantity,
        l_extendedprice,
        int(shipdate_lo_days),
        int(shipdate_hi_days),
        float(discount_lo),
        float(discount_hi),
        float(quantity_hi),
    )
    return torch.tensor(revenue, dtype=torch.float64, device=l_extendedprice.device)


def _query_core(
    l_shipdate_days: torch.Tensor,
    l_discount: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    discount_lo: float,
    discount_hi: float,
    quantity_hi: float,
) -> torch.Tensor:
    """
    Runs the hot path in CUDA.

    This replaces eager PyTorch boolean-mask creation, gather, multiply, and sum
    with one fused CUDA scan. The kernel uses coalesced loads, nested conditional
    loads for low-selectivity predicates, shared-memory block reductions, and one
    global atomic add per block.
    """
    return _launch_q6_revenue(
        l_shipdate_days,
        l_discount,
        l_quantity,
        l_extendedprice,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
    discount_lo: float = 0.05,
    discount_hi: float = 0.07,
    quantity_hi: float = 24.0,
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= CAST('1994-01-01' AS date)
          AND l_shipdate < CAST('1995-01-01' AS date)
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24;

    Description:
        Computes TPC-H Q6 revenue. cuDF is used only for column access and the
        required conversion of l_shipdate from datetime64[s] to int32 epoch days.
        The bulk scan runs in a hand-written CUDA kernel that fuses filter,
        projection, and aggregation. This is faster for the memory-bound hot path
        because it avoids PyTorch boolean-mask temporaries, avoids filtered
        gathers, performs conditional loads of later columns, and reduces with
        shared memory plus one atomicAdd per block.

    Params:
        tables (dict[str, cudf.DataFrame]): Input tables; uses tables["lineitem"].
        shipdate_lo (str): Inclusive lower ship-date bound.
        shipdate_hi (str): Exclusive upper ship-date bound.
        discount_lo (float): Inclusive lower discount bound.
        discount_hi (float): Inclusive upper discount bound.
        quantity_hi (float): Exclusive quantity bound.
    """
    df_lineitem = tables["lineitem"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    result = _EXT.q6_revenue_launcher(
        l_shipdate_days_t,
        l_discount_t,
        l_quantity_t,
        l_extendedprice_t,
        int(shipdate_lo_days),
        int(shipdate_hi_days),
        float(discount_lo),
        float(discount_hi),
        float(quantity_hi),
    )
    return float(result)
