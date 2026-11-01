import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA kernel source – fused filter (shipdate) + projection + two reductions
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// Block‑level reduction for two double accumulators
__device__ __forceinline__ void block_reduce(double *shared_tot,
                                            double *shared_promo,
                                            int tid) {
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_tot[tid]   += shared_tot[tid + stride];
            shared_promo[tid] += shared_promo[tid + stride];
        }
        __syncthreads();
    }
}

// ----------------------------------------------------------------------
// Kernel: one pass over the (joined & pre‑filtered) table
// ----------------------------------------------------------------------
__global__ void promo_rev_kernel(
    const double *__restrict__ extprice,   // l_extendedprice
    const double *__restrict__ discount,   // l_discount
    const int32_t *__restrict__ shipdate, // l_shipdate (days)
    const uint8_t *__restrict__ promo,    // p_type_is_promo (0/1)
    int32_t date_lo,
    int32_t date_hi,
    double *__restrict__ out_total,        // global accumulator for total revenue
    double *__restrict__ out_promo,        // global accumulator for promo revenue
    int64_t N) {

    double my_total = 0.0;
    double my_promo = 0.0;

    int64_t idx = blockIdx.x * (int64_t)BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        int32_t sd = shipdate[idx];
        // shipdate filter (already pre‑filtered, but kept for correctness)
        if (sd >= date_lo && sd < date_hi) {
            double rev = extprice[idx] * (1.0 - discount[idx]);
            my_total = rev;
            if (promo[idx]) {
                my_promo = rev;
            }
        }
    }

    // shared memory for block reduction
    __shared__ double block_total[BLOCK_SIZE];
    __shared__ double block_promo[BLOCK_SIZE];
    block_total[threadIdx.x] = my_total;
    block_promo[threadIdx.x] = my_promo;

    // intra‑block reduction
    block_reduce(block_total, block_promo, threadIdx.x);

    // atomic add once per block
    if (threadIdx.x == 0) {
        atomicAdd(out_total, block_total[0]);
        atomicAdd(out_promo, block_promo[0]);
    }
}

// ----------------------------------------------------------------------
// Launcher exposed via pybind11
// ----------------------------------------------------------------------
double promo_rev_launcher(
    torch::Tensor extprice,   // double, CUDA, contiguous
    torch::Tensor discount,   // double, CUDA, contiguous
    torch::Tensor shipdate,   // int32, CUDA, contiguous
    torch::Tensor promo,      // uint8, CUDA, contiguous
    int32_t date_lo,
    int32_t date_hi) {

    TORCH_CHECK(extprice.dtype() == torch::kFloat64, "extprice must be float64");
    TORCH_CHECK(discount.dtype() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(shipdate.dtype() == torch::kInt32,   "shipdate must be int32");
    TORCH_CHECK(promo.dtype()    == torch::kUInt8,   "promo must be uint8");
    TORCH_CHECK(extprice.is_cuda() && discount.is_cuda()
                && shipdate.is_cuda() && promo.is_cuda(),
                "All tensors must be on CUDA");

    // ensure contiguity
    extprice = extprice.contiguous();
    discount = discount.contiguous();
    shipdate = shipdate.contiguous();
    promo    = promo.contiguous();

    const int64_t N = extprice.numel();

    // device scalars for the two aggregates
    auto total_buf = torch::zeros({1}, extprice.options());
    auto promo_buf = torch::zeros({1}, extprice.options());

    const int blocks = static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    promo_rev_kernel<<<blocks, BLOCK_SIZE>>>(
        extprice.data_ptr<double>(),
        discount.data_ptr<double>(),
        shipdate.data_ptr<int32_t>(),
        promo.data_ptr<uint8_t>(),
        date_lo,
        date_hi,
        total_buf.data_ptr<double>(),
        promo_buf.data_ptr<double>(),
        N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA launch error: ")
                                 + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    double total = total_buf.item<double>();
    double promo_val = promo_buf.item<double>();

    if (total == 0.0) {
        return 0.0;
    }
    return 100.0 * promo_val / total;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("promo_rev_launcher", &promo_rev_launcher,
          "Fused filter/project/reduce for TPC‑H promo revenue (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Compile / load the CUDA extension (once per interpreter)
# ----------------------------------------------------------------------
def _load_cuda_extension():
    """Write the CUDA source to a temporary file and compile it."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q19_cuda_")
    src_path = os.path.join(tmpdir, "promo_rev.cu")
    with open(src_path, "w") as f:
        f.write(_CUDA_SOURCE)

    # No explicit build_directory – let torch manage the temporary location.
    return load(
        name="tpch_q19_cuda_ext",
        sources=[src_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        verbose=False,
        with_cuda=True,
    )

_EXT = _load_cuda_extension()


# ----------------------------------------------------------------------
# Core GPU routine – thin wrapper around the compiled launcher
# ----------------------------------------------------------------------
def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    p_type_is_promo: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> torch.Tensor:
    """
    GPU‑accelerated core for Q19.

    All tensors must be CUDA tensors with:
        - l_extendedprice, l_discount : torch.float64
        - l_shipdate_days           : torch.int32
        - p_type_is_promo           : torch.uint8 (0/1)

    Returns a 0‑dim torch.Tensor containing the final float64 result.
    """
    result = _EXT.promo_rev_launcher(
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        p_type_is_promo,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return torch.tensor(result, dtype=torch.float64, device=l_extendedprice.device)


# ----------------------------------------------------------------------
# Public API – unchanged signature, but with pre‑filtering to reduce join size
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL:
        SELECT 100.00 *
               SUM(CASE WHEN p_type LIKE 'PROMO%' THEN
                           l_extendedprice * (1 - l_discount)
                       ELSE 0 END) /
               SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
        FROM lineitem, part
        WHERE l_partkey = p_partkey
          AND l_shipdate >= DATE '1995-09-01'
          AND l_shipdate < DATE '1995-10-01';

    Description:
        The ``lineitem`` table is first filtered on ``l_shipdate`` using cuDF,
        dramatically reducing the number of rows that participate in the
        join with ``part``.  After the join only the four columns needed for the
        aggregation are materialised and transferred to CUDA tensors.
        A single hand‑written CUDA kernel then fuses the (remaining) shipdate
        filter, revenue projection and the two reductions (total and promotional
        revenue) into one memory‑bound pass, yielding a measurable speed‑up
        over the reference eager/compile plans.

    Params:
        shipdate_lo (str): inclusive lower bound on ``l_shipdate`` (YYYY‑MM‑DD).
        shipdate_hi (str): exclusive upper bound on ``l_shipdate`` (YYYY‑MM‑DD).

    Returns:
        float: promo_revenue as defined by the SQL query.
    """
    # --------------------------------------------------------------
    # Convert date strings to epoch days (int32) for the kernel
    # --------------------------------------------------------------
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # --------------------------------------------------------------
    # Pre‑filter lineitem on shipdate using cuDF (GPU‑accelerated)
    # --------------------------------------------------------------
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_ts = pd.Timestamp(shipdate_lo)
    shipdate_hi_ts = pd.Timestamp(shipdate_hi)

    lineitem_mask = (df_lineitem["l_shipdate"] >= shipdate_lo_ts) & (
        df_lineitem["l_shipdate"] < shipdate_hi_ts
    )
    filtered_lineitem = df_lineitem[lineitem_mask]

    # --------------------------------------------------------------
    # Prepare the part side: boolean flag for PROMO types
    # --------------------------------------------------------------
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    # --------------------------------------------------------------
    # Join the filtered lineitem with the projected part table
    # --------------------------------------------------------------
    joined = filtered_lineitem.merge(
        part_proj,
        left_on="l_partkey",
        right_on="p_partkey",
        how="inner",
    )

    # --------------------------------------------------------------
    # Transfer required columns to CUDA tensors
    # --------------------------------------------------------------
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"]
          .astype("int64") // 86400)
         .astype("int32"))
        .to_cupy(),
        device="cuda"
    )

    l_extendedprice_t = torch.as_tensor(
        joined["l_extendedprice"].to_cupy(),
        dtype=torch.float64,
        device="cuda"
    )

    l_discount_t = torch.as_tensor(
        joined["l_discount"].to_cupy(),
        dtype=torch.float64,
        device="cuda"
    )

    # Boolean flag as uint8 for the kernel
    p_type_is_promo_t = torch.as_tensor(
        joined["p_type_is_promo"]
            .astype("bool")
            .to_cupy(),
        dtype=torch.uint8,
        device="cuda"
    )

    # --------------------------------------------------------------
    # Execute fused CUDA kernel
    # --------------------------------------------------------------
    result_tensor = _query_core(
        l_extendedprice_t,
        l_discount_t,
        l_shipdate_days_t,
        p_type_is_promo_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )
    return float(result_tensor.item())
