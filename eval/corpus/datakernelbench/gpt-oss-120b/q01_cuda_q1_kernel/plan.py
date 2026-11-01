"""
Optimized CUDA implementation of TPC‑H Q1.

SQL:
SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity) AS sum_qty,
    sum(l_extendedprice) AS sum_base_price,
    sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    avg(l_quantity) AS avg_qty,
    avg(l_extendedprice) AS avg_price,
    avg(l_discount) AS avg_disc,
    count(*) AS count_order
FROM lineitem
WHERE l_shipdate <= CAST('1998-09-02' AS date)
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;

Description:
* cuDF is used only for cheap string‑handling (categorical encoding/decoding).
* All heavy work – filter, projection, and six aggregations – is fused
  into a single CUDA kernel (`q1_kernel`).  The kernel reads each row once,
  evaluates the ship‑date predicate, computes the derived columns
  (`disc_price`, `charge`) and atomically accumulates per‑group totals.
* Because the group‑by keys have at most 256 × 256 distinct values,
  we allocate fixed‑size global accumulator arrays (size 65536) and use
  `atomicAdd` for double‑precision sums and a casted `unsigned long long`
  atomic for the count.  This avoids multiple kernel launches and
  dramatically reduces memory traffic compared with the reference
  eager‑Torch plan (which materialises many intermediate tensors).

Params:
    tables (dict[str, cudf.DataFrame]): Must contain key "lineitem".
    shipdate_hi (str): Upper bound for l_shipdate (inclusive).

Returns:
    cudf.DataFrame with the ten columns listed in the SELECT, ordered by
    (l_returnflag, l_linestatus).
"""

import os
import tempfile

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source – a single string written to a temporary .cu file at import.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define MAX_GROUPS (256 * 256)   // rf_code * 256 + ls_code

// ------------------------------------------------------------------
// One‑pass kernel: filter + projection + group‑by aggregation.
// ------------------------------------------------------------------
__global__ void q1_kernel(
    const int32_t* __restrict__ rf_code,          // l_returnflag categorical code
    const int32_t* __restrict__ ls_code,          // l_linestatus categorical code
    const double*  __restrict__ qty,              // l_quantity
    const double*  __restrict__ price,            // l_extendedprice
    const double*  __restrict__ disc,             // l_discount
    const double*  __restrict__ tax,              // l_tax
    const int32_t* __restrict__ shipdate_days,    // l_shipdate (days since epoch)
    const int32_t shipdate_hi,                    // filter bound
    double* __restrict__ sum_qty,
    double* __restrict__ sum_base_price,
    double* __restrict__ sum_disc_price,
    double* __restrict__ sum_charge,
    double* __restrict__ sum_disc,
    unsigned long long* __restrict__ cnt,        // count per group
    const int64_t n                               // number of rows
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Date predicate (<= shipdate_hi)
    int32_t sday = shipdate_days[idx];
    if (sday > shipdate_hi) return;

    // Compute group index
    int32_t grp = rf_code[idx] * 256 + ls_code[idx];
    // Load required columns
    double q   = qty[idx];
    double p   = price[idx];
    double d   = disc[idx];
    double t   = tax[idx];

    double disc_price = p * (1.0 - d);
    double charge     = disc_price * (1.0 + t);

    // Atomic accumulation (FP64 atomics are supported on sm_90)
    atomicAdd(&sum_qty[grp],          q);
    atomicAdd(&sum_base_price[grp],   p);
    atomicAdd(&sum_disc_price[grp],   disc_price);
    atomicAdd(&sum_charge[grp],       charge);
    atomicAdd(&sum_disc[grp],         d);
    atomicAdd(&cnt[grp], (unsigned long long)1);
}

// ------------------------------------------------------------------
// Host‑side launcher called from Python.
// ------------------------------------------------------------------
void q1_launcher(
    torch::Tensor rf_code,          // int32
    torch::Tensor ls_code,          // int32
    torch::Tensor qty,              // float64
    torch::Tensor price,            // float64
    torch::Tensor disc,             // float64
    torch::Tensor tax,              // float64
    torch::Tensor shipdate_days,    // int32
    const int32_t shipdate_hi,
    torch::Tensor sum_qty,          // float64, size MAX_GROUPS
    torch::Tensor sum_base_price,   // float64, size MAX_GROUPS
    torch::Tensor sum_disc_price,   // float64, size MAX_GROUPS
    torch::Tensor sum_charge,       // float64, size MAX_GROUPS
    torch::Tensor sum_disc,         // float64, size MAX_GROUPS
    torch::Tensor cnt               // int64, size MAX_GROUPS
) {
    TORCH_CHECK(rf_code.is_cuda(), "rf_code must be on CUDA");
    TORCH_CHECK(ls_code.is_cuda(), "ls_code must be on CUDA");
    TORCH_CHECK(qty.is_cuda(),     "qty must be on CUDA");
    TORCH_CHECK(price.is_cuda(),   "price must be on CUDA");
    TORCH_CHECK(disc.is_cuda(),    "disc must be on CUDA");
    TORCH_CHECK(tax.is_cuda(),     "tax must be on CUDA");
    TORCH_CHECK(shipdate_days.is_cuda(), "shipdate_days must be on CUDA");
    TORCH_CHECK(sum_qty.is_cuda() && sum_qty.numel() == MAX_GROUPS, "sum_qty size mismatch");
    TORCH_CHECK(sum_base_price.is_cuda() && sum_base_price.numel() == MAX_GROUPS, "sum_base_price size mismatch");
    TORCH_CHECK(sum_disc_price.is_cuda() && sum_disc_price.numel() == MAX_GROUPS, "sum_disc_price size mismatch");
    TORCH_CHECK(sum_charge.is_cuda() && sum_charge.numel() == MAX_GROUPS, "sum_charge size mismatch");
    TORCH_CHECK(sum_disc.is_cuda() && sum_disc.numel() == MAX_GROUPS, "sum_disc size mismatch");
    TORCH_CHECK(cnt.is_cuda() && cnt.numel() == MAX_GROUPS, "cnt size mismatch");

    int64_t n = qty.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    q1_kernel<<<blocks, threads>>>(
        rf_code.data_ptr<int32_t>(),
        ls_code.data_ptr<int32_t>(),
        qty.data_ptr<double>(),
        price.data_ptr<double>(),
        disc.data_ptr<double>(),
        tax.data_ptr<double>(),
        shipdate_days.data_ptr<int32_t>(),
        shipdate_hi,
        sum_qty.data_ptr<double>(),
        sum_base_price.data_ptr<double>(),
        sum_disc_price.data_ptr<double>(),
        sum_charge.data_ptr<double>(),
        sum_disc.data_ptr<double>(),
        reinterpret_cast<unsigned long long*>(cnt.data_ptr<int64_t>()),
        n
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch error: ", cudaGetErrorString(err));
    }
    // Synchronize to make results visible to host
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q1_launcher", &q1_launcher, "TPC‑H Q1 fused aggregation (CUDA)");
}
'''

# ----------------------------------------------------------------------
# Helper: compile the CUDA source once at import time.
# ----------------------------------------------------------------------
def _load_cuda_extension() -> object:
    """Compile the CUDA kernels and return the loaded Python extension."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q1_cuda_")
    cu_path = os.path.join(tmpdir, "q1_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="tpch_q1_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )


# Load (or rebuild) the extension
_EXT = _load_cuda_extension()

# ----------------------------------------------------------------------
# Core fused aggregation using the compiled CUDA kernel.
# ----------------------------------------------------------------------
_MAX_GROUPS = 256 * 256  # enough for rf_code*256 + ls_code


def _query_core(
    l_returnflag_code: torch.Tensor,
    l_linestatus_code: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_tax: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    shipdate_hi_days: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """Execute the fused kernel and post‑process the result.

    Returns tensors in the order required by the reference plan.
    """
    device = l_quantity.device

    # Allocate per‑group accumulator tensors (size = MAX_GROUPS)
    sum_qty = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    sum_base_price = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    sum_disc_price = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    sum_charge = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    sum_disc = torch.zeros(_MAX_GROUPS, dtype=torch.float64, device=device)
    count = torch.zeros(_MAX_GROUPS, dtype=torch.int64, device=device)

    # Launch the CUDA kernel
    _EXT.q1_launcher(
        l_returnflag_code,
        l_linestatus_code,
        l_quantity,
        l_extendedprice,
        l_discount,
        l_tax,
        l_shipdate_days,
        shipdate_hi_days,
        sum_qty,
        sum_base_price,
        sum_disc_price,
        sum_charge,
        sum_disc,
        count,
    )

    # Keep only groups that actually appear (count > 0)
    mask = count > 0
    active_keys = torch.nonzero(mask, as_tuple=False).squeeze(1)  # shape (n_groups,)

    # Decode returnflag / linestatus codes
    rf_out = (active_keys // 256).to(torch.int64)
    ls_out = (active_keys % 256).to(torch.int64)

    # Gather aggregates for active groups
    sum_qty = sum_qty[active_keys]
    sum_base_price = sum_base_price[active_keys]
    sum_disc_price = sum_disc_price[active_keys]
    sum_charge = sum_charge[active_keys]
    sum_disc = sum_disc[active_keys]
    count_order = count[active_keys]

    # Compute averages (float64)
    count_f = count_order.to(torch.float64)
    avg_qty = sum_qty / count_f
    avg_price = sum_base_price / count_f
    avg_disc = sum_disc / count_f

    # Order by (rf, ls) – which is exactly the integer key order
    order = torch.argsort(active_keys)

    return (
        rf_out[order],
        ls_out[order],
        sum_qty[order],
        sum_base_price[order],
        sum_disc_price[order],
        sum_charge[order],
        avg_qty[order],
        avg_price[order],
        avg_disc[order],
        count_order[order],
    )


# ----------------------------------------------------------------------
# Public API – unchanged signature from the reference implementation.
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_hi: str = "1998-09-02",
) -> cudf.DataFrame:
    """
    SQL: SELECT l_returnflag, l_linestatus,
         sum(l_quantity) AS sum_qty,
         sum(l_extendedprice) AS sum_base_price,
         sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
         sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
         avg(l_quantity) AS avg_qty,
         avg(l_extendedprice) AS avg_price,
         avg(l_discount) AS avg_disc,
         count(*) AS count_order
         FROM lineitem
         WHERE l_shipdate <= CAST('1998-09-02' AS date)
         GROUP BY l_returnflag, l_linestatus
         ORDER BY l_returnflag, l_linestatus;
    Description:
        Performs categorical encoding of the two string columns with cuDF,
        moves numeric columns to CUDA tensors, runs a single fused CUDA kernel
        that computes all six aggregates and the count, then decodes the
        categorical codes back to strings and assembles the final DataFrame.
    Params:
        tables (dict[str, cudf.DataFrame]): Must contain a "lineitem" DataFrame.
        shipdate_hi (str): Upper‑bound ship date (inclusive) in YYYY‑MM‑DD format.
    Returns:
        cudf.DataFrame with ten columns matching the SELECT order.
    """
    df_lineitem = tables["lineitem"]

    # Convert upper bound to integer days since epoch
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Categorical encoding of string columns (returnflag, linestatus)
    rf_series = df_lineitem["l_returnflag"].astype("str")
    ls_series = df_lineitem["l_linestatus"].astype("str")

    rf_codes = rf_series.astype("category")
    ls_codes = ls_series.astype("category")

    # Preserve category ordering for later decoding
    rf_categories = rf_codes.cat.categories.to_pandas().tolist()
    ls_categories = ls_codes.cat.categories.to_pandas().tolist()

    # Move all needed columns to CUDA tensors (float64 for numeric cols)
    l_returnflag_code_t = torch.as_tensor(
        rf_codes.cat.codes.astype("int32").to_cupy()
    )
    l_linestatus_code_t = torch.as_tensor(
        ls_codes.cat.codes.astype("int32").to_cupy()
    )
    l_quantity_t = torch.as_tensor(
        df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64
    )
    l_extendedprice_t = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        df_lineitem["l_discount"].to_cupy(), dtype=torch.float64
    )
    l_tax_t = torch.as_tensor(
        df_lineitem["l_tax"].to_cupy(), dtype=torch.float64
    )
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400)
         .astype("int32"))
        .to_cupy()
    )

    # Run the fused CUDA aggregation
    (
        rf_out,
        ls_out,
        sum_qty,
        sum_base_price,
        sum_disc_price,
        sum_charge,
        avg_qty,
        avg_price,
        avg_disc,
        count_order,
    ) = _query_core(
        l_returnflag_code_t,
        l_linestatus_code_t,
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_tax_t,
        l_shipdate_days_t,
        shipdate_hi_days,
    )

    # Decode categorical codes back to original strings
    rf_strings = [rf_categories[i] for i in rf_out.cpu().tolist()]
    ls_strings = [ls_categories[i] for i in ls_out.cpu().tolist()]

    # Assemble result DataFrame
    return cudf.DataFrame(
        {
            "l_returnflag": cudf.Series(rf_strings),
            "l_linestatus": cudf.Series(ls_strings),
            "sum_qty": sum_qty,
            "sum_base_price": sum_base_price,
            "sum_disc_price": sum_disc_price,
            "sum_charge": sum_charge,
            "avg_qty": avg_qty,
            "avg_price": avg_price,
            "avg_disc": avg_disc,
            "count_order": count_order,
        }
    )
