import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <algorithm>

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
    
    // Grid-stride loop indexing
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    double my_rev = 0.0;
    
    // Late materialization / predicate pushdown: 
    // Load expensive columns only when previous predicates succeed.
    for (int64_t i = idx; i < n_elements; i += stride) {
        int32_t sd = shipdate_ptr[i];
        if (sd >= date_lo && sd < date_hi) {
            double disc = discount_ptr[i];
            if (disc >= disc_lo && disc <= disc_hi) {
                double qty = quantity_ptr[i];
                if (qty < qty_hi) {
                    my_rev += price_ptr[i] * disc;
                }
            }
        }
    }
    
    // Block-level reduction in shared memory
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Single atomicAdd per block
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
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64);
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32);
    
    // Move to CUDA if needed
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
    
    // Ensure contiguous memory for pointers
    price = price.contiguous();
    discount = discount.contiguous();
    quantity = quantity.contiguous();
    shipdate_days = shipdate_days.contiguous();
    
    int64_t n = price.numel();
    auto out = torch::zeros({1}, price.options());
    
    // Configure grid dimensions
    int blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 4096);
    
    if (n > 0) {
        q6_kernel<<<blocks, BLOCK_SIZE>>>(
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            quantity.data_ptr<double>(),
            shipdate_days.data_ptr<int32_t>(),
            date_lo, date_hi, disc_lo, disc_hi, qty_hi,
            out.data_ptr<double>(), n
        );
    }
    
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q6_launcher", &q6_launcher, "TPC-H Q6 revenue (CUDA)");
}
'''


def _load_cuda():
    """Compiles and loads the CUDA kernel as a PyTorch C++ extension."""
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
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )


# Compile kernel exactly once on import
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
         
    Description: Computes the forecast revenue increase from eliminating discounts in a given range for a year.
                 The filter and aggregate are fused into a single memory-bound CUDA kernel for optimal performance.
                 Predicate pushdown (late materialization) is used in the kernel to heavily reduce memory bandwidth.
                 
    Params:
        shipdate_lo (str): inclusive lower bound ship date.
        shipdate_hi (str): exclusive upper bound ship date.
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
    """
    df_lineitem = tables["lineitem"]

    # Convert bounding dates to epoch days
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Convert target columns to Torch device tensors
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    # Launch the fused kernel
    res = _EXT.q6_launcher(
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
    
    return float(res)
