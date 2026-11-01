import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA kernel source (fused filter + projection + sum reduction)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// One‑dimensional kernel: each thread evaluates one row, builds local revenue,
// then a shared‑memory tree reduction yields one block‑level sum which is
// atomically added to the global output.
__global__ void q6_fused_kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ quantity,
    const int32_t* __restrict__ shipdate,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi,
    double* __restrict__ out,
    int64_t N)
{
    __shared__ double shmem[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_val = 0.0;

    if (idx < N) {
        int32_t sd = shipdate[idx];
        double d  = discount[idx];
        double q  = quantity[idx];
        bool keep = (sd >= date_lo) && (sd < date_hi) &&
                    (d  >= disc_lo) && (d <= disc_hi) &&
                    (q  < qty_hi);
        if (keep) {
            my_val = price[idx] * d;
        }
    }

    shmem[threadIdx.x] = my_val;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // One atomic add per block
    if (threadIdx.x == 0) {
        atomicAdd(out, shmem[0]);
    }
}

// Launcher called from Python. All tensors must be contiguous and on CUDA.
// Returns the scalar revenue as a double.
double q6_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor quantity,
    torch::Tensor shipdate,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi)
{
    // Basic type checks – we expect double columns and int32 shipdate.
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(shipdate.scalar_type() == torch::kInt32,   "shipdate must be int32");

    // Ensure device residency
    if (!price.is_cuda())   price = price.cuda();
    if (!discount.is_cuda())discount = discount.cuda();
    if (!quantity.is_cuda())quantity = quantity.cuda();
    if (!shipdate.is_cuda())shipdate = shipdate.cuda();

    // Contiguity for raw pointer access
    price    = price.contiguous();
    discount = discount.contiguous();
    quantity = quantity.contiguous();
    shipdate = shipdate.contiguous();

    const int64_t N = price.numel();
    const int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Output buffer – a single double in GPU memory
    auto out = torch::zeros({1}, price.options());

    // Kernel launch
    q6_fused_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        quantity.data_ptr<double>(),
        shipdate.data_ptr<int32_t>(),
        date_lo, date_hi,
        disc_lo, disc_hi,
        qty_hi,
        out.data_ptr<double>(),
        N
    );
    // Synchronize to make sure the kernel finished before reading.
    cudaDeviceSynchronize();

    return out.item<double>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q6_launcher", &q6_launcher, "TPC‑H Q6 revenue (CUDA fused kernel)");
}
'''

def _load_cuda_extension() -> torch.nn.Module:
    """
    Compiles the CUDA source embedded in ``_CUDA_SOURCE`` and returns the loaded
    Python extension module. The compilation is performed once per process.
    """
    tmpdir = tempfile.mkdtemp(prefix="q6_cuda_")
    cu_path = os.path.join(tmpdir, "q6_kernel.cu")
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
        verbose=False,
    )

# Load (or re‑load) the compiled extension
_EXT = _load_cuda_extension()

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
    """
    Invoke the fused CUDA kernel and wrap the returned double in a 0‑dim
    ``torch.Tensor`` (float64). Using a tensor keeps the same return type as the
    reference implementation.
    """
    revenue = _EXT.q6_launcher(
        l_extendedprice,
        l_discount,
        l_quantity,
        l_shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
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
    Core computation delegated to the fused CUDA kernel. Mirrors the reference
    eager version but runs entirely on the GPU in a single pass, eliminating
    intermediate masks, indexing, and multiple memory traversals.
    """
    return _launch_q6(
        l_extendedprice,
        l_discount,
        l_quantity,
        l_shipdate_days,
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
        SELECT SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01'
          AND l_shipdate <  DATE '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24;

    Description:
        The pipeline pulls the required columns from the cuDF ``lineitem`` table,
        casts the ship‑date column to *int32* epoch‑days, and moves all columns
        to GPU tensors (float64 for monetary fields). The heavy‑weight work—
        filtering, multiplying, and summing—is performed by a single CUDA kernel,
        which fuses the three operations into one memory‑bound pass and reduces
        per‑block results in shared memory before a single atomic addition per
        block, dramatically lowering global‑memory traffic compared to the
        reference eager torch implementation.

    Params:
        shipdate_lo (str): inclusive lower bound ship date (YYYY‑MM‑DD).
        shipdate_hi (str): exclusive upper bound ship date (YYYY‑MM‑DD).
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
    Returns:
        float: revenue (scalar) matching the reference plan.
    """
    df_lineitem = tables["lineitem"]

    # Convert date bounds to int32 days since epoch.
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Transfer required columns to CUDA tensors.
    l_extendedprice_t = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        df_lineitem["l_discount"].to_cupy(), dtype=torch.float64
    )
    l_quantity_t = torch.as_tensor(
        df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64
    )
    # Shipdate: datetime64[s] → int64 epoch seconds → int32 days.
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"]
          .astype("int64") // 86400)
         .astype("int32"))
        .to_cupy(),
        dtype=torch.int32,
    )

    # Execute fused kernel.
    result_tensor = _query_core(
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
    # Convert 0‑dim tensor to Python float.
    return float(result_tensor.item())
