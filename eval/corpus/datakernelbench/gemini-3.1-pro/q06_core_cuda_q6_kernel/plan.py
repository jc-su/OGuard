import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q6_kernel(
    const int32_t* __restrict__ shipdate_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ price_ptr,
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
    int64_t stride = blockDim.x * gridDim.x;
    
    double my_rev = 0.0;
    
    // Grid-stride loop handles arbitrary sizes efficiently
    for (int64_t i = idx; i < n_elements; i += stride) {
        int32_t sd = shipdate_ptr[i];
        // Short-circuiting evaluation saves memory bandwidth
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
    
    block_revenue[threadIdx.x] = my_rev;
    __syncthreads();
    
    // Parallel reduction in shared memory
    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // One atomic add per block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}

torch::Tensor q6_launcher(
    torch::Tensor shipdate,
    torch::Tensor discount,
    torch::Tensor quantity,
    torch::Tensor price,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64);
    
    shipdate = shipdate.contiguous();
    discount = discount.contiguous();
    quantity = quantity.contiguous();
    price = price.contiguous();
    
    int64_t n = price.numel();
    auto out = torch::zeros({1}, price.options());
    
    if (n > 0) {
        int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // Cap the number of blocks, relying on grid-stride loop
        if (blocks > 65536) blocks = 65536;
        
        q6_kernel<<<blocks, BLOCK_SIZE>>>(
            shipdate.data_ptr<int32_t>(),
            discount.data_ptr<double>(),
            quantity.data_ptr<double>(),
            price.data_ptr<double>(),
            date_lo, date_hi, disc_lo, disc_hi, qty_hi,
            out.data_ptr<double>(), n
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q6_launcher", &q6_launcher, "TPC-H Q6 revenue (CUDA)");
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
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

# Compile once at module import
_EXT = _load_cuda()

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
    Computes Q6 revenue in a single fused CUDA kernel.
    Runs entirely on GPU, avoiding multiple intermediate tensors and kernel launches.
    Lazy memory loading significantly saves memory bandwidth.
    """
    return _EXT.q6_launcher(
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
