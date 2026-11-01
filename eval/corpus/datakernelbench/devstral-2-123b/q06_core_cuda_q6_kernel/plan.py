import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#define BLOCK_SIZE 1024

__global__ void q6_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    double date_lo,
    double date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    if (idx < n_elements) {
        double sd = static_cast<double>(shipdate_ptr[idx]);
        double disc = discount_ptr[idx];
        double qty = quantity_ptr[idx];
        bool keep = (sd >= date_lo) && (sd < date_hi)
                 && (disc >= disc_lo) && (disc <= disc_hi)
                 && (qty < qty_hi);
        if (keep)
            my_rev = price_ptr[idx] * disc;
    }
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out_ptr, block_revenue[0]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", [](torch::Tensor price,
                       torch::Tensor discount,
                       torch::Tensor quantity,
                       torch::Tensor shipdate_days,
                       double date_lo,
                       double date_hi,
                       double disc_lo,
                       double disc_hi,
                       double qty_hi) -> double {
        TORCH_CHECK(price.scalar_type() == torch::kFloat64);
        if (!price.is_cuda()) price = price.cuda();
        if (!discount.is_cuda()) discount = discount.cuda();
        if (!quantity.is_cuda()) quantity = quantity.cuda();
        if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
        price = price.contiguous();
        discount = discount.contiguous();
        quantity = quantity.contiguous();
        shipdate_days = shipdate_days.contiguous();
        int64_t n = price.numel();
        auto out = torch::zeros({1}, price.options());
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        q6_kernel<<<blocks, BLOCK_SIZE>>>(
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            quantity.data_ptr<double>(),
            shipdate_days.data_ptr<int32_t>(),
            date_lo, date_hi, disc_lo, disc_hi, qty_hi,
            out.data_ptr<double>(), n
        );
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        return out.item<double>();
    }, "Q6 fusedlaunch");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q6_cuda_")
    cu_path = os.path.join(tmpdir, "_q6_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q6_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90", "-gencode", "arch=compute_90,code=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=True,
    )

_EXT = _load_cuda()

def _launch_q6(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_quantity: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    discount_lo: float,
    discount_hi: float,
    quantity_hi: float,
) -> torch.Tensor:
    rev = _EXT.forward(
        l_extendedprice, l_discount, l_quantity, l_shipdate_days,
        float(shipdate_lo_days), float(shipdate_hi_days),
        discount_lo, discount_hi, quantity_hi,
    )
    return torch.tensor(rev, dtype=torch.float64, device=l_extendedprice.device)

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
    return _launch_q6(
        l_extendedprice, l_discount, l_quantity, l_shipdate_days,
        shipdate_lo_days, shipdate_hi_days, discount_lo, discount_hi, quantity_hi,
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
    SQL: SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem
         WHERE l_shipdate >= CAST('1994-01-01' AS date) AND l_shipdate < CAST('1995-01-01' AS date)
         AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;
    Description: Computes the forecast revenue increase from eliminating discounts in a given range for a year.
    Params:
        shipdate_lo (str): inclusive lower bound ship date.
        shipdate_hi (str): exclusive upper bound ship date.
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
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

    res = _query_core(
        l_shipdate_days_t,
        l_discount_t,
        l_quantity_t,
        l_extendedprice_t,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
    )
    return float(res.item())
