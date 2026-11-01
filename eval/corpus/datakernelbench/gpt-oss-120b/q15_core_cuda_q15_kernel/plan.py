"""
Optimized CUDA implementation of TPC-H Q15 core compute.

Public API:
    _query_core(...): same signature as reference, returns (sel_idx, sel_keys, sel_rev).
    run_query(...): copied verbatim from reference (unchanged).

The hot path (filter → per‑supplier revenue aggregation) is fused into a single
CUDA kernel that:
    * Scans lineitem columns coalesced.
    * Applies the shipdate predicate in registers.
    * Computes revenue = l_extendedprice * (1 - l_discount).
    * Binary‑searches the sorted supplier key list (unique) to map each row
      to a supplier index.
    * Atomically adds revenue and a row count to per‑supplier buffers.
A block‑level reduction is not needed because each supplier accumulator is
updated via atomicAdd; the number of suppliers (≈100 k) is small enough for
fast atomics on H100.

After the kernel, the host unsorts the per‑supplier totals back to the original
supplier order, builds the `present` mask from the row‑count buffer, finds the
maximum revenue and selects all suppliers attaining it, preserving the reference
ordering semantics.

Only the compute kernel is replaced; all I/O and DataFrame handling remain
identical to the reference plan.
"""
import os
import tempfile

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source – placed in a string, written to a temporary file and loaded.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Binary search for key in a sorted int64 array.
// Returns position of the first element >= key (i.e. lower_bound).
__device__ __forceinline__ int64_t lower_bound(const int64_t* data,
                                               int64_t size,
                                               int64_t key) {
    int64_t lo = 0;
    int64_t hi = size;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        int64_t val = data[mid];
        if (val < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void q15_kernel(const int64_t* __restrict__ suppkey,
                           const double* __restrict__ price,
                           const double* __restrict__ disc,
                           const int32_t* __restrict__ shipdate,
                           const int64_t* __restrict__ sorted_supp,
                           double* __restrict__ totals,
                           int64_t* __restrict__ counts,
                           int32_t date_lo,
                           int32_t date_hi,
                           int64_t n_rows,
                           int64_t n_sup) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    // Apply shipdate filter.
    int32_t sd = shipdate[idx];
    if (sd < date_lo || sd >= date_hi) return;

    // Compute revenue for this lineitem row.
    double rev = price[idx] * (1.0 - disc[idx]);

    // Map l_suppkey to supplier index via binary search.
    int64_t key = suppkey[idx];
    int64_t pos = lower_bound(sorted_supp, n_sup, key);
    if (pos < n_sup && sorted_supp[pos] == key) {
        // Atomically accumulate revenue and row count.
        atomicAdd(&totals[pos], rev);
        // atomicAdd for signed 64‑bit uses unsigned long long.
        atomicAdd((unsigned long long*)&counts[pos], 1ULL);
    }
}

// Launcher --------------------------------------------------------------
void q15_launcher(torch::Tensor suppkey,
                  torch::Tensor price,
                  torch::Tensor disc,
                  torch::Tensor shipdate,
                  torch::Tensor sorted_supp,
                  torch::Tensor totals,
                  torch::Tensor counts,
                  int32_t date_lo,
                  int32_t date_hi) {
    TORCH_CHECK(suppkey.is_cuda() && price.is_cuda() && disc.is_cuda() && shipdate.is_cuda(),
                "All input tensors must be CUDA tensors");
    TORCH_CHECK(sorted_supp.is_cuda() && totals.is_cuda() && counts.is_cuda(),
                "Output tensors must be CUDA tensors");
    TORCH_CHECK(suppkey.dtype() == torch::kInt64, "l_suppkey must be int64");
    TORCH_CHECK(price.dtype() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(disc.dtype() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(shipdate.dtype() == torch::kInt32, "l_shipdate_days must be int32");
    TORCH_CHECK(sorted_supp.dtype() == torch::kInt64, "sorted_suppkey must be int64");
    TORCH_CHECK(totals.dtype() == torch::kFloat64, "totals must be float64");
    TORCH_CHECK(counts.dtype() == torch::kInt64, "counts must be int64");

    // Ensure contiguous memory.
    suppkey = suppkey.contiguous();
    price   = price.contiguous();
    disc    = disc.contiguous();
    shipdate = shipdate.contiguous();
    sorted_supp = sorted_supp.contiguous();
    totals = totals.contiguous();
    counts = counts.contiguous();

    int64_t n_rows = suppkey.numel();
    int64_t n_sup  = sorted_supp.numel();

    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>((n_rows + threads - 1) / threads);

    q15_kernel<<<blocks, threads>>>(
        suppkey.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        disc.data_ptr<double>(),
        shipdate.data_ptr<int32_t>(),
        sorted_supp.data_ptr<int64_t>(),
        totals.data_ptr<double>(),
        counts.data_ptr<int64_t>(),
        date_lo,
        date_hi,
        n_rows,
        n_sup
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
    // Synchronize to make results visible to the host.
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", &q15_launcher,
          "TPC-H Q15 fused aggregation kernel (CUDA)");
}
'''

def _load_cuda():
    """
    Write the CUDA source to a temporary file and compile it with
    `torch.utils.cpp_extension.load`. The compiled module is cached for the
    lifetime of the Python process.
    """
    tmpdir = tempfile.mkdtemp(prefix="q15_cuda_")
    cu_path = os.path.join(tmpdir, "q15_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q15_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        verbose=False,
        with_cuda=True,
        build_directory=build_dir,
    )

# Compile / load once at import time.
_EXT = _load_cuda()

def _launch_q15(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    sorted_suppkey: torch.Tensor,
    totals: torch.Tensor,
    counts: torch.Tensor,
    shipdate_lo: int,
    shipdate_hi: int,
) -> None:
    """
    Invoke the fused CUDA kernel that computes per‑supplier revenue
    (`totals`) and row counts (`counts`).

    All tensors must reside on the same CUDA device.
    """
    _EXT.q15_launcher(
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        sorted_suppkey,
        totals,
        counts,
        shipdate_lo,
        shipdate_hi,
    )

def _query_core(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Core compute for TPC‑H Q15.

    Steps:
      1. Sort supplier keys (required for binary search).
      2. Allocate per‑supplier buffers (totals, counts).
      3. Launch the fused CUDA kernel that:
         – filters by shipdate,
         – computes revenue per lineitem,
         – maps each row to a supplier via binary search,
         – atomically accumulates revenue and row count.
      4. Un‑sort the results back to the original supplier order.
      5. Derive `present` mask from the count buffer, find the max revenue,
         and return the indices/keys/revenues of the top supplier(s).
    """
    # 1. Sort supplier keys and keep the permutation to restore original order.
    sorted_suppkey, sort_idx = torch.sort(s_suppkey)

    n_sup = s_suppkey.numel()
    device = s_suppkey.device

    # 2. Allocate output buffers.
    totals_sorted = torch.zeros(n_sup, dtype=torch.float64, device=device)
    counts_sorted = torch.zeros(n_sup, dtype=torch.int64, device=device)

    # 3. Run the fused kernel.
    _launch_q15(
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        sorted_suppkey,
        totals_sorted,
        counts_sorted,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # 4. Un‑sort totals back to original supplier order.
    totals = torch.empty_like(totals_sorted)
    totals[sort_idx] = totals_sorted

    # Build present mask: supplier contributed at least one row.
    present_sorted = counts_sorted > 0
    present = torch.empty_like(present_sorted)
    present[sort_idx] = present_sorted

    # 5. Determine max revenue among present suppliers.
    max_rev = totals[present].max()
    sel_mask = present & (totals == max_rev)

    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    sel_keys = s_suppkey[sel_idx]

    # Order result by supplier key as required by the SQL.
    order = torch.argsort(sel_keys)
    sel_idx = sel_idx[order]

    return sel_idx, s_suppkey[sel_idx], totals[sel_idx]

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1996-01-01",
    shipdate_hi: str = "1996-04-01",
) -> cudf.DataFrame:
    """
    SQL:
      WITH revenue AS (
        SELECT l_suppkey AS supplier_no,
               SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
        GROUP BY l_suppkey
      )
      SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
      FROM supplier, revenue
      WHERE s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
      ORDER BY s_suppkey;
    Description: TPC-H Q15 — find supplier(s) with the top revenue in a given quarter.
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_price_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_disc_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)

    sel_idx_t, sel_keys_t, sel_rev_t = _query_core(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        s_suppkey_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # Use indices to gather string/phone columns from cuDF supplier table
    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_t),
    })
