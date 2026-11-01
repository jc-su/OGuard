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

__global__ void filter_aggregate_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_price,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate,
    int32_t date_lo,
    int32_t date_hi,
    const int64_t* __restrict__ sorted_suppkeys,
    double* __restrict__ out_revenue,
    int64_t n_lineitem,
    int n_supplier
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lineitem) return;

    int32_t shipdate = l_shipdate[idx];
    if (shipdate >= date_lo && shipdate < date_hi) {
        int64_t suppkey = l_suppkey[idx];
        // binary search in sorted supplier keys
        int64_t lo = 0, hi = n_supplier;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (sorted_suppkeys[mid] < suppkey) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < n_supplier && sorted_suppkeys[lo] == suppkey) {
            double revenue = l_price[idx] * (1.0 - l_discount[idx]);
            atomicAdd(&out_revenue[lo], revenue);
        }
    }
}

void filter_aggregate(
    torch::Tensor l_suppkey,
    torch::Tensor l_price,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate,
    int32_t date_lo,
    int32_t date_hi,
    torch::Tensor sorted_suppkeys,
    torch::Tensor out_revenue
) {
    TORCH_CHECK(l_suppkey.scalar_type() == torch::kInt64, "l_suppkey must be int64");
    TORCH_CHECK(l_price.scalar_type() == torch::kFloat64, "l_price must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(l_shipdate.scalar_type() == torch::kInt32, "l_shipdate must be int32");
    TORCH_CHECK(sorted_suppkeys.scalar_type() == torch::kInt64, "sorted_suppkeys must be int64");
    TORCH_CHECK(out_revenue.scalar_type() == torch::kFloat64, "out_revenue must be float64");
    TORCH_CHECK(out_revenue.is_cuda(), "out_revenue must be a CUDA tensor");

    // ensure contiguity
    l_suppkey = l_suppkey.contiguous();
    l_price = l_price.contiguous();
    l_discount = l_discount.contiguous();
    l_shipdate = l_shipdate.contiguous();
    sorted_suppkeys = sorted_suppkeys.contiguous();
    out_revenue = out_revenue.contiguous();

    const int64_t n_lineitem = l_suppkey.size(0);
    const int n_supplier = sorted_suppkeys.size(0);
    const int grid = (n_lineitem + BLOCK_SIZE - 1) / BLOCK_SIZE;

    filter_aggregate_kernel<<<grid, BLOCK_SIZE>>>(
        l_suppkey.data_ptr<int64_t>(),
        l_price.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate.data_ptr<int32_t>(),
        date_lo,
        date_hi,
        sorted_suppkeys.data_ptr<int64_t>(),
        out_revenue.data_ptr<double>(),
        n_lineitem,
        n_supplier
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_aggregate", &filter_aggregate, "Filter lineitems and aggregate revenue by supplier (CUDA)");
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
    Description: TPC-H Q15 — find supplier(s) with the top revenue in a given quarter. This implementation
      uses a fused CUDA kernel to filter lineitems, compute revenue, and aggregate per supplier in a single
      pass. The per‑supplier totals are then unsorted and processed on the CPU to find the maximum revenue
      and select the matching suppliers.
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    # Convert date bounds to integer epoch days (int32)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Prepare GPU tensors for lineitem columns
    l_suppkey_t = torch.tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64, device='cuda')
    l_price_t = torch.tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64, device='cuda')
    l_disc_t = torch.tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64, device='cuda')
    l_shipdate_days_t = torch.tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32, device='cuda'
    )

    # Sort supplier keys to map supplier -> index
    s_suppkey_t = torch.tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64, device='cuda')
    sorted_s, sort_idx = torch.sort(s_suppkey_t)
    n_sup = s_suppkey_t.numel()

    # Output buffer for revenue per supplier (sorted order)
    revenue_sorted = torch.zeros(n_sup, dtype=torch.float64, device='cuda')

    # Launch fused CUDA kernel
    _EXT.filter_aggregate(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        shipdate_lo_days,
        shipdate_hi_days,
        sorted_s,
        revenue_sorted,
    )

    # Unsort totals back to original supplier order
    totals = torch.empty_like(revenue_sorted)
    totals[sort_idx] = revenue_sorted

    # Determine suppliers that have any revenue
    present = totals != 0
    if not present.any():
        # No matching lineitems, return empty result with correct schema
        return cudf.DataFrame({
            "s_suppkey": cudf.Series(dtype="int64"),
            "s_name": cudf.Series(dtype="object"),
            "s_address": cudf.Series(dtype="object"),
            "s_phone": cudf.Series(dtype="object"),
            "total_revenue": cudf.Series(dtype="float64"),
        })

    # Find maximum revenue among present suppliers
    max_rev = totals[present].max()
    sel_mask = present & (totals == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

    # Order selected suppliers by s_suppkey
    order = torch.argsort(s_suppkey_t[sel_idx])
    sel_idx = sel_idx[order]

    # Gather supplier details using the selected indices
    sel_idx_np = sel_idx.cpu().numpy()
    supplier_sel = df_supplier.iloc[sel_idx_np].reset_index(drop=True)

    # Build final DataFrame
    result = cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(totals[sel_idx]),
    })
    return result
