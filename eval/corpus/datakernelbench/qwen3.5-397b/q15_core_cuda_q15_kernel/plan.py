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

__global__ void q15_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_price,
    const double* __restrict__ l_disc,
    const int32_t* __restrict__ l_shipdate,
    const int64_t* __restrict__ supp_pos,
    double* __restrict__ block_totals,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_li,
    int64_t n_sup,
    int64_t block_stride
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    double* my_block_totals = block_totals + (int64_t)blockIdx.x * block_stride;

    for (int64_t idx = tid; idx < n_li; idx += stride) {
        int32_t sd = l_shipdate[idx];
        if (sd < date_lo || sd >= date_hi) continue;

        int64_t pos = supp_pos[idx];
        if (pos < 0) continue;
        
        double rev = l_price[idx] * (1.0 - l_disc[idx]);
        atomicAdd(&my_block_totals[pos], rev);
    }
}

torch::Tensor q15_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_price,
    torch::Tensor l_disc,
    torch::Tensor l_shipdate,
    torch::Tensor supp_pos,
    int32_t date_lo,
    int32_t date_hi,
    int num_blocks
) {
    TORCH_CHECK(l_suppkey.is_cuda(), "l_suppkey must be CUDA");
    TORCH_CHECK(l_price.is_cuda(), "l_price must be CUDA");
    TORCH_CHECK(l_disc.is_cuda(), "l_disc must be CUDA");
    TORCH_CHECK(l_shipdate.is_cuda(), "l_shipdate must be CUDA");
    TORCH_CHECK(supp_pos.is_cuda(), "supp_pos must be CUDA");
    
    TORCH_CHECK(l_suppkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(l_price.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_disc.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_shipdate.scalar_type() == torch::kInt32);
    TORCH_CHECK(supp_pos.scalar_type() == torch::kInt64);

    l_suppkey = l_suppkey.contiguous();
    l_price = l_price.contiguous();
    l_disc = l_disc.contiguous();
    l_shipdate = l_shipdate.contiguous();
    supp_pos = supp_pos.contiguous();

    int64_t n_li = l_suppkey.numel();
    int64_t n_sup = supp_pos.max().item<int64_t>() + 1;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(l_suppkey.device());
    torch::Tensor block_totals = torch::zeros({num_blocks, n_sup}, options);
    
    int64_t block_stride = n_sup;
    
    q15_kernel<<<num_blocks, BLOCK_SIZE>>>(
        l_suppkey.data_ptr<int64_t>(),
        l_price.data_ptr<double>(),
        l_disc.data_ptr<double>(),
        l_shipdate.data_ptr<int32_t>(),
        supp_pos.data_ptr<int64_t>(),
        block_totals.data_ptr<double>(),
        date_lo,
        date_hi,
        n_li,
        n_sup,
        block_stride
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return block_totals;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", &q15_launcher, "TPC-H Q15 CUDA Kernel");
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
    Computes the top revenue supplier(s) using a fused CUDA kernel for the heavy scan/aggregation.
    
    GPU Plan:
    1. Sort s_suppkey on GPU (PyTorch) to enable binary search.
    2. Pre-compute position mapping for all lineitem rows via searchsorted (once, outside kernel).
    3. Launch CUDA kernel with grid-stride loop to scan lineitem.
       - Filter by date (cheap comparison).
       - Lookup pre-computed supplier position (O(1)).
       - Accumulate revenue into per-block per-supplier buffer (reduces atomic contention).
    4. Reduce per-block buffers to final totals (PyTorch sum).
    5. Unsort totals and select max revenue suppliers (PyTorch).
    
    This avoids binary search in the hot kernel path and reduces global atomic contention.
    """
    # Sort supplier keys for binary search
    sorted_s, sort_idx = torch.sort(s_suppkey)
    n_sup = s_suppkey.numel()
    
    # Pre-compute position mapping for all lineitem rows (outside kernel)
    pos = torch.searchsorted(sorted_s, l_suppkey)
    # Mark invalid positions (supplier not in table)
    valid = (pos < n_sup) & (sorted_s[pos.clamp(max=n_sup - 1)] == l_suppkey)
    supp_pos = torch.where(valid, pos, torch.tensor(-1, dtype=torch.int64, device=pos.device))
    
    num_blocks = 1024  # Reduced grid size for H100
    
    # Launch CUDA kernel to compute per-block per-supplier totals
    block_totals = _EXT.q15_launcher(
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        supp_pos,
        shipdate_lo_days,
        shipdate_hi_days,
        num_blocks
    )
    
    # Reduce block totals to final per-supplier totals (in sorted order)
    totals_sorted = block_totals.sum(dim=0, dtype=torch.float64)
    
    # Unsort back to original s_suppkey order
    totals = torch.empty_like(totals_sorted)
    totals[sort_idx] = totals_sorted
    
    # Identify present suppliers (totals > 0 implies presence)
    present_sorted = totals_sorted > 0.0
    present = torch.empty_like(present_sorted)
    present[sort_idx] = present_sorted
    
    # Max revenue over present suppliers
    if present.any():
        max_rev = totals[present].max()
    else:
        max_rev = torch.tensor(0.0, dtype=torch.float64, device=totals.device)
    
    # Select suppliers equal to max revenue
    sel_mask = present & (totals == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    
    # Order by s_suppkey
    if sel_idx.numel() > 0:
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
