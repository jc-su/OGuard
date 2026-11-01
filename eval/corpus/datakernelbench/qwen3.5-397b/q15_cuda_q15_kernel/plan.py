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

__device__ int binary_search(const long long* arr, int n, long long val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void q15_kernel(
    const long long* l_suppkey,
    const double* l_price,
    const double* l_disc,
    const int* l_shipdate,
    int shipdate_lo,
    int shipdate_hi,
    const long long* s_suppkey_sorted,
    int n_suppliers,
    double* revenue_out,
    int n_lineitems
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lineitems) return;

    int sd = l_shipdate[idx];
    if (sd < shipdate_lo || sd >= shipdate_hi) return;

    long long supp = l_suppkey[idx];
    double rev = l_price[idx] * (1.0 - l_disc[idx]);

    int pos = binary_search(s_suppkey_sorted, n_suppliers, supp);
    if (pos < n_suppliers && s_suppkey_sorted[pos] == supp) {
        atomicAdd(&revenue_out[pos], rev);
    }
}

torch::Tensor q15_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_price,
    torch::Tensor l_disc,
    torch::Tensor l_shipdate,
    int shipdate_lo,
    int shipdate_hi,
    torch::Tensor s_suppkey_sorted,
    int n_suppliers
) {
    TORCH_CHECK(l_suppkey.scalar_type() == torch::kInt64, "l_suppkey must be int64");
    TORCH_CHECK(l_price.scalar_type() == torch::kFloat64, "l_price must be float64");
    TORCH_CHECK(l_disc.scalar_type() == torch::kFloat64, "l_disc must be float64");
    TORCH_CHECK(l_shipdate.scalar_type() == torch::kInt32, "l_shipdate must be int32");
    TORCH_CHECK(s_suppkey_sorted.scalar_type() == torch::kInt64, "s_suppkey_sorted must be int64");

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto revenue_out = torch::zeros({n_suppliers}, options);

    int n = l_suppkey.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const long long* l_suppkey_ptr = static_cast<const long long*>(l_suppkey.data_ptr());
    const double* l_price_ptr = static_cast<const double*>(l_price.data_ptr());
    const double* l_disc_ptr = static_cast<const double*>(l_disc.data_ptr());
    const int* l_shipdate_ptr = static_cast<const int*>(l_shipdate.data_ptr());
    const long long* s_suppkey_ptr = static_cast<const long long*>(s_suppkey_sorted.data_ptr());
    double* revenue_ptr = static_cast<double*>(revenue_out.data_ptr());

    q15_kernel<<<blocks, BLOCK_SIZE>>>(
        l_suppkey_ptr,
        l_price_ptr,
        l_disc_ptr,
        l_shipdate_ptr,
        shipdate_lo, shipdate_hi,
        s_suppkey_ptr,
        n_suppliers,
        revenue_ptr,
        n
    );

    cudaDeviceSynchronize();
    return revenue_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", &q15_launcher, "TPC-H Q15 Revenue Aggregation");
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
    # Sort supplier keys to enable binary search in kernel
    sorted_s, sort_idx = torch.sort(s_suppkey)
    n_sup = sorted_s.numel()
    
    # Ensure all tensors are contiguous and on CUDA
    l_suppkey = l_suppkey.contiguous().cuda()
    l_extendedprice = l_extendedprice.contiguous().cuda()
    l_discount = l_discount.contiguous().cuda()
    l_shipdate_days = l_shipdate_days.contiguous().cuda()
    sorted_s = sorted_s.contiguous().cuda()
    
    # Launch fused CUDA kernel: filter + project + scatter-add
    # This fuses the date filter, revenue calculation, and group-by aggregation
    # into a single pass over lineitem, minimizing HBM reads.
    revenue_sorted = _EXT.q15_launcher(
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        sorted_s,
        n_sup,
    )
    
    # Unsort revenue to match original s_suppkey order
    revenue = torch.empty_like(revenue_sorted)
    revenue[sort_idx] = revenue_sorted
    
    # Identify present suppliers (revenue > 0)
    present = revenue > 0
    
    # Handle case where no suppliers match
    if not present.any():
        return torch.empty(0, dtype=torch.int64, device=s_suppkey.device), \
               torch.empty(0, dtype=torch.int64, device=s_suppkey.device), \
               torch.empty(0, dtype=torch.float64, device=s_suppkey.device)
    
    max_rev = revenue[present].max()
    sel_mask = present & (revenue == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    
    # Order by s_suppkey
    sel_keys = s_suppkey[sel_idx]
    order = torch.argsort(sel_keys)
    sel_idx = sel_idx[order]
    
    return sel_idx, s_suppkey[sel_idx], revenue[sel_idx]

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
                 Uses fused CUDA kernel for filter/project/aggregate to minimize HBM passes.
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

    if len(sel_idx_t) == 0:
        return cudf.DataFrame({
            "s_suppkey": cudf.Series([], dtype="int64"),
            "s_name": cudf.Series([], dtype="object"),
            "s_address": cudf.Series([], dtype="object"),
            "s_phone": cudf.Series([], dtype="object"),
            "total_revenue": cudf.Series([], dtype="float64"),
        })

    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_t),
    })
