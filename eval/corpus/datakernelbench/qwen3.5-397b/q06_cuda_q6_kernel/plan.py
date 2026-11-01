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
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    if (idx < n_elements) {
        int32_t sd = shipdate_ptr[idx];
        double disc = discount_ptr[idx];
        double qty = quantity_ptr[idx];
        // Predicate: shipdate in [lo, hi), discount in [lo, hi], quantity < hi
        if (sd >= date_lo && sd < date_hi &&
            disc >= disc_lo && disc <= disc_hi &&
            qty < qty_hi) {
            my_rev = price_ptr[idx] * disc;
        }
    }
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    // Block reduction (tree)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}

double q6_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor quantity,
    torch::Tensor shipdate_days,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi
) {
    // Ensure inputs are on CUDA and contiguous
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
    
    price = price.contiguous();
    discount = discount.contiguous();
    quantity = quantity.contiguous();
    shipdate_days = shipdate_days.contiguous();

    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32, "shipdate must be int32");

    int64_t n = price.numel();
    auto out = torch::zeros({1}, price.options()); // float64, cuda
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q6_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        quantity.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        date_lo, date_hi, disc_lo, disc_hi, qty_hi,
        out.data_ptr<double>(),
        n
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q6_launcher", &q6_launcher, "TPC-H Q6 revenue (CUDA fused kernel)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q6_cuda_")
    cu_path = os.path.join(tmpdir, "_q6_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    
    # H100 is sm_90
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
    Description: Computes forecast revenue increase. Uses fused CUDA kernel for filter/project/aggregate to minimize HBM passes (memory-bound optimization).
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables.
        shipdate_lo (str): Inclusive lower bound ship date.
        shipdate_hi (str): Exclusive upper bound ship date.
        discount_lo (float): Inclusive lower bound discount.
        discount_hi (float): Inclusive upper bound discount.
        quantity_hi (float): Exclusive upper bound quantity.
    """
    df_lineitem = tables["lineitem"]

    # Date bounds: int32 epoch days
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Prepare tensors (GPU zero-copy via cupy interface)
    # l_shipdate: datetime64[s] -> int64 seconds -> int32 days
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    # Launch fused kernel
    rev = _EXT.q6_launcher(
        l_extendedprice_t,
        l_discount_t,
        l_quantity_t,
        l_shipdate_days_t,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
    )
    return float(rev)
