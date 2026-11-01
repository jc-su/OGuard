import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA kernel source (fusion of filter → projection → sum)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// fused kernel for TPC-H Q6
__global__ void q6_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi,
    double* __restrict__ out_ptr,
    int64_t n_elements)
{
    __shared__ double block_sum[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_val = 0.0;

    if (idx < n_elements) {
        int32_t sd   = shipdate_ptr[idx];
        double  disc = discount_ptr[idx];
        double  qty  = quantity_ptr[idx];
        bool keep = (sd   >= date_lo) && (sd   <  date_hi) &&
                    (disc >= disc_lo) && (disc <= disc_hi) &&
                    (qty  <  qty_hi);
        if (keep) {
            my_val = price_ptr[idx] * disc;
        }
    }

    block_sum[threadIdx.x] = my_val;
    __syncthreads();

    // reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // one atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_sum[0]);
    }
}

// launcher returning a double scalar
double q6_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor quantity,
    torch::Tensor shipdate_days,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi)
{
    // ensure correct datatype
    TORCH_CHECK(price.scalar_type() == torch::kFloat64,
                "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64,
                "discount must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64,
                "quantity must be float64");
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32,
                "shipdate_days must be int32");

    // make sure tensors are on CUDA and contiguous
    if (!price.is_cuda())      price = price.cuda();
    if (!discount.is_cuda())   discount = discount.cuda();
    if (!quantity.is_cuda())   quantity = quantity.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();

    price = price.contiguous();
    discount = discount.contiguous();
    quantity = quantity.contiguous();
    shipdate_days = shipdate_days.contiguous();

    const int64_t n = price.numel();
    auto out = torch::zeros({1}, price.options());

    const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q6_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        quantity.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        date_lo, date_hi,
        disc_lo, disc_hi,
        qty_hi,
        out.data_ptr<double>(),
        n
    );

    // synchronize to catch launch errors early
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch error: ") + cudaGetErrorString(err));
    }

    // return the scalar value
    return out.item<double>();
}

// PyBind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q6_launcher", &q6_launcher, "TPC-H Q6 fused kernel (CUDA)");
}
'''

def _load_cuda_extension() -> object:
    """Compile the CUDA source at import time and return the loaded module."""
    tmpdir = tempfile.mkdtemp(prefix="q6_cuda_")
    cu_path = os.path.join(tmpdir, "q6_kernel.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    # H100 → sm_90
    return load(
        name="q6_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

# Load (compile once per interpreter session)
_EXT = _load_cuda_extension()


def _launch_q6(
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
    Thin wrapper around the compiled CUDA launcher.
    Returns a scalar torch.Tensor on the same device as ``l_extendedprice``.
    """
    rev = _EXT.q6_launcher(
        l_extendedprice,
        l_discount,
        l_quantity,
        l_shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        float(discount_lo),
        float(discount_hi),
        float(quantity_hi),
    )
    # ``rev`` is a Python float (double) – wrap as a GPU tensor
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
    """
    GPU‑accelerated core for TPC‑H Q6.
    Mirrors the reference signature but executes a fused CUDA kernel
    that filters, multiplies and reduces in a single pass, reducing
    memory traffic and eliminating per‑row Python overhead.
    """
    return _launch_q6(
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

    # Convert cuDF columns to GPU tensors
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
