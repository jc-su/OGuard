import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <stdint.h>

#define BLOCK_SIZE 1024

__global__ void q15_filter_aggregate_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    const int64_t* __restrict__ sorted_suppkeys,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ out_totals,
    int64_t n_rows,
    int64_t n_suppliers
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    int32_t shipdate = l_shipdate_days[idx];
    if (shipdate < date_lo || shipdate >= date_hi) return;

    double revenue = l_extendedprice[idx] * (1.0 - l_discount[idx]);
    int64_t supp = l_suppkey[idx];

    // binary search lower_bound in sorted_suppkeys
    int64_t lo = 0, hi = n_suppliers;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (sorted_suppkeys[mid] < supp) lo = mid + 1;
        else hi = mid;
    }
    int64_t pos = lo;
    if (pos < n_suppliers && sorted_suppkeys[pos] == supp) {
        atomicAdd(&out_totals[pos], revenue);
    }
}

void q15_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    torch::Tensor sorted_suppkeys,
    int32_t date_lo,
    int32_t date_hi,
    torch::Tensor out_totals
) {
    TORCH_CHECK(l_suppkey.is_cuda(), "l_suppkey must be a CUDA tensor");
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be a CUDA tensor");
    TORCH_CHECK(l_discount.is_cuda(), "l_discount must be a CUDA tensor");
    TORCH_CHECK(l_shipdate_days.is_cuda(), "l_shipdate_days must be a CUDA tensor");
    TORCH_CHECK(sorted_suppkeys.is_cuda(), "sorted_suppkeys must be a CUDA tensor");
    TORCH_CHECK(out_totals.is_cuda(), "out_totals must be a CUDA tensor");

    // Ensure contiguous
    l_suppkey = l_suppkey.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    l_shipdate_days = l_shipdate_days.contiguous();
    sorted_suppkeys = sorted_suppkeys.contiguous();
    out_totals = out_totals.contiguous();

    auto n_rows = l_suppkey.size(0);
    auto n_suppliers = sorted_suppkeys.size(0);

    int grid = (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q15_filter_aggregate_kernel<<<grid, BLOCK_SIZE>>>(
        l_suppkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate_days.data_ptr<int32_t>(),
        sorted_suppkeys.data_ptr<int64_t>(),
        date_lo,
        date_hi,
        out_totals.data_ptr<double>(),
        n_rows,
        n_suppliers
    );

    // Synchronize to ensure kernel completion before returning
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", &q15_launcher, "Q15 filter + aggregate kernel");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q15_cuda_")
    cu_path = os.path.join(tmpdir, "_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q15_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

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
    Core compute for Q15: filter lineitem rows by shipdate, compute revenue,
    aggregate per supplier, and find supplier(s) with maximum revenue.
    Returns (selected_indices, selected_suppkey, selected_total_revenue).
    """
    device = s_suppkey.device
    # Ensure all inputs are on GPU (is_cuda is a property, not a method)
    if not l_suppkey.is_cuda:
        l_suppkey = l_suppkey.to(device)
    if not l_extendedprice.is_cuda:
        l_extendedprice = l_extendedprice.to(device)
    if not l_discount.is_cuda:
        l_discount = l_discount.to(device)
    if not l_shipdate_days.is_cuda:
        l_shipdate_days = l_shipdate_days.to(device)

    # Sort supplier keys to obtain mapping to index space
    sorted_s, sort_idx = torch.sort(s_suppkey)
    n_sup = s_suppkey.numel()

    # Allocate per-supplier accumulator (in sorted order)
    totals_sorted = torch.zeros(n_sup, dtype=torch.float64, device=device)

    # Launch fused CUDA kernel: filter + project + aggregate
    _EXT.q15_launcher(
        l_suppkey.contiguous(),
        l_extendedprice.contiguous(),
        l_discount.contiguous(),
        l_shipdate_days.contiguous(),
        sorted_s.contiguous(),
        shipdate_lo_days,
        shipdate_hi_days,
        totals_sorted,
    )
    # Ensure kernel finished before we read the result
    torch.cuda.synchronize()

    # Unsort to original supplier order
    totals = torch.empty_like(totals_sorted)
    totals[sort_idx] = totals_sorted

    # Presence: any supplier that received revenue (>0)
    present = totals > 0
    if not present.any():
        # No matching rows – return empty result
        empty_int64 = torch.tensor([], dtype=torch.int64, device=device)
        empty_float64 = torch.tensor([], dtype=torch.float64, device=device)
        return empty_int64, empty_int64, empty_float64

    # Maximum revenue among present suppliers
    max_rev = totals[present].max()

    # Select suppliers with revenue equal to max
    sel_mask = present & (totals == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

    # Order by supplier key
    sel_keys = s_suppkey[sel_idx]
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
