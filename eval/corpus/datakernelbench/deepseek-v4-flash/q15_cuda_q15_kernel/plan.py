import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

// Fused filter + project + group-by kernel for TPC-H Q15.
// Each thread processes one lineitem row: checks date window, computes revenue,
// binary-searches sorted supplier keys, and atomically adds to per-supplier total.
__global__ void q15_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    const int64_t* __restrict__ l_suppkey_ptr,
    const int64_t* __restrict__ sorted_supplier_ptr,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_elements,
    int64_t n_suppliers,
    double* __restrict__ out_totals
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    int32_t sd = shipdate_ptr[idx];
    if (sd >= date_lo && sd < date_hi) {
        double price_v = price_ptr[idx];
        double disc_v = discount_ptr[idx];
        double revenue = price_v * (1.0 - disc_v);
        
        // Binary search in sorted supplier keys
        int64_t target = l_suppkey_ptr[idx];
        const int64_t* lo = sorted_supplier_ptr;
        const int64_t* hi = sorted_supplier_ptr + n_suppliers;
        while (lo < hi) {
            const int64_t* mid = lo + ((hi - lo) >> 1);
            if (*mid < target) lo = mid + 1;
            else hi = mid;
        }
        
        if (lo < sorted_supplier_ptr + n_suppliers && *lo == target) {
            int64_t supplier_idx = (int64_t)(lo - sorted_supplier_ptr);
            atomicAdd(&out_totals[supplier_idx], revenue);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", [](
        torch::Tensor price,
        torch::Tensor discount,
        torch::Tensor shipdate_days,
        torch::Tensor l_suppkey,
        torch::Tensor sorted_supplier,
        int32_t date_lo,
        int32_t date_hi,
        torch::Tensor out_totals
    ) {
        TORCH_CHECK(price.scalar_type() == torch::kFloat64);
        TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
        TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32);
        TORCH_CHECK(l_suppkey.scalar_type() == torch::kInt64);
        TORCH_CHECK(sorted_supplier.scalar_type() == torch::kInt64);
        TORCH_CHECK(out_totals.scalar_type() == torch::kFloat64);
        
        int64_t n = price.numel();
        int64_t n_sup = sorted_supplier.numel();
        
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        q15_kernel<<<blocks, BLOCK_SIZE>>>(
            price.data_ptr<double>(),
            discount.data_ptr<double>(),
            shipdate_days.data_ptr<int32_t>(),
            l_suppkey.data_ptr<int64_t>(),
            sorted_supplier.data_ptr<int64_t>(),
            date_lo,
            date_hi,
            n,
            n_sup,
            out_totals.data_ptr<double>()
        );
        
        cudaDeviceSynchronize();
    }, "TPC-H Q15: filter + project + group-by revenue");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q15_cuda_")
    cu_path = os.path.join(tmpdir, "_q15_kernels.cu")
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
    Description: TPC-H Q15 — find supplier(s) with the top revenue in a given quarter.
      Uses a fused CUDA kernel for filter/project/group-by. Post-processing with PyTorch.
    Params:
      tables: dict of cudf DataFrames (keys: 'lineitem', 'supplier')
      shipdate_lo: inclusive lower bound date for filtering lineitem
      shipdate_hi: exclusive upper bound date for filtering lineitem
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]
    
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)
    
    # Convert to GPU tensors
    l_suppkey = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64).cuda()
    l_price = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64).cuda()
    l_discount = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64).cuda()
    l_shipdate_days = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32
    ).cuda()
    
    s_suppkey = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64).cuda()
    
    # Sort supplier keys for binary search in CUDA kernel
    sorted_supplier, sort_idx = torch.sort(s_suppkey)
    
    # Allocate output tensor (per-supplier totals, initialized to zero)
    n_suppliers = len(s_suppkey)
    out_totals = torch.zeros(n_suppliers, dtype=torch.float64, device="cuda")
    
    # Launch fused CUDA kernel
    _EXT.q15_launcher(
        l_price, l_discount, l_shipdate_days, l_suppkey,
        sorted_supplier,
        shipdate_lo_days, shipdate_hi_days,
        out_totals
    )
    
    # Un-sort totals back to original s_suppkey order
    totals_original = torch.empty_like(out_totals)
    totals_original[sort_idx] = out_totals
    
    # Identify suppliers with non-zero revenue (present in the date window)
    present = totals_original > 0.0
    
    if not present.any():
        return cudf.DataFrame({
            "s_suppkey": cudf.Series([], dtype="int64"),
            "s_name": cudf.Series([], dtype="object"),
            "s_address": cudf.Series([], dtype="object"),
            "s_phone": cudf.Series([], dtype="object"),
            "total_revenue": cudf.Series([], dtype="float64"),
        })
    
    max_rev = totals_original[present].max()
    
    # Select suppliers with max revenue
    sel_mask = present & (totals_original == max_rev)
    sel_indices = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    
    # Order by s_suppkey
    sel_keys = s_suppkey[sel_indices]
    order = torch.argsort(sel_keys)
    sel_indices = sel_indices[order].contiguous()
    
    # Move to CPU for cuDF indexing
    sel_indices_cpu = sel_indices.cpu()
    sel_rev_cpu = totals_original[sel_indices].cpu()
    
    sel_idx_series = cudf.Series(sel_indices_cpu.numpy())
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)
    
    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_cpu.numpy()),
    })
