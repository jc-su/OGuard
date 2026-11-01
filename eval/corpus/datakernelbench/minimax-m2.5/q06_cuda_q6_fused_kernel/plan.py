import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q6_fused_kernel(
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
    // Shared memory for block-level reduction
    __shared__ double block_revenue[BLOCK_SIZE];
    
    // Calculate global thread index
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Per-thread partial revenue accumulator
    double my_rev = 0.0;
    
    // Each thread processes one element (coalesced memory access)
    if (idx < n_elements) {
        // Load data with restrict pointers for better compiler optimization
        const int32_t sd = shipdate_ptr[idx];
        const double disc = discount_ptr[idx];
        const double qty = quantity_ptr[idx];
        
        // Predicate evaluation in registers
        const bool keep = (sd >= date_lo) && (sd < date_hi)
                       && (disc >= disc_lo) && (disc <= disc_hi)
                       && (qty < qty_hi);
        
        // Compute revenue only for matching rows
        if (keep) {
            my_rev = price_ptr[idx] * disc;
        }
    }
    
    // Store per-thread result to shared memory
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Block-level tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block atomically adds to global output
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
    // Validate input dtypes (must be float64 for decimal columns)
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32, "shipdate_days must be int32");
    
    // Ensure tensors are on CUDA and contiguous
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();
    if (!shipdate_days.is_cuda()) shipdate_days = shipdate_days.cuda();
    
    price = price.contiguous();
    discount = discount.contiguous();
    quantity = quantity.contiguous();
    shipdate_days = shipdate_days.contiguous();
    
    // Calculate grid dimensions
    const int64_t n = price.numel();
    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate output tensor on GPU
    auto out = torch::zeros({1}, price.options().device(torch::kCUDA));
    
    // Launch fused kernel: filter + project + aggregate in one pass
    q6_fused_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        quantity.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        date_lo,
        date_hi,
        disc_lo,
        disc_hi,
        qty_hi,
        out.data_ptr<double>(),
        n
    );
    
    // Synchronize and return result
    cudaDeviceSynchronize();
    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q6_launcher", &q6_launcher, "TPC-H Q6: fused filter-project-aggregate revenue (CUDA)");
}
'''

def _load_cuda():
    """
    Load CUDA extension with the fused Q6 kernel.
    Writes kernel source to temp file and JIT-compiles via nvcc.
    Targets sm_90 (H100) for optimal performance.
    """
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

# Compile once at module import
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
    
    Description: Computes the forecast revenue increase from eliminating discounts
    in a given range for a year. Uses a fused CUDA kernel that performs filter,
    projection, and aggregation in a single pass for maximum memory bandwidth
    utilization on H100.
    
    Params:
        shipdate_lo (str): inclusive lower bound ship date.
        shipdate_hi (str): exclusive upper bound ship date.
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
    """
    df_lineitem = tables["lineitem"]
    
    # Convert date bounds to int32 epoch days (same as reference)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)
    
    # Transfer cuDF columns to torch tensors on GPU
    # Using float64 for decimal columns per reference convention
    price = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(),
        dtype=torch.float64
    )
    discount = torch.as_tensor(
        df_lineitem["l_discount"].to_cupy(),
        dtype=torch.float64
    )
    quantity = torch.as_tensor(
        df_lineitem["l_quantity"].to_cupy(),
        dtype=torch.float64
    )
    shipdate_days = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    
    # Launch fused CUDA kernel: one pass for filter + project + aggregate
    # Uses shared memory reduction and one atomicAdd per block
    rev = _EXT.q6_launcher(
        price,
        discount,
        quantity,
        shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
    )
    
    return float(rev)
