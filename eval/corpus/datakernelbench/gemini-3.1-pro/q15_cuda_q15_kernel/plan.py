import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void q15_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_price,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate,
    int32_t date_lo,
    int32_t date_hi,
    double* __restrict__ out_revenue,
    int32_t* __restrict__ out_present,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t sd = l_shipdate[idx];
        if (sd >= date_lo && sd < date_hi) {
            int64_t suppkey = l_suppkey[idx];
            double rev = l_price[idx] * (1.0 - l_discount[idx]);
            // Global atomicAdd. Because selectivity is low (1 quarter out of 7 years = ~3.5%)
            // and spread over up to 100,000 supplier keys, collisions are minimal and 
            // global atomics are very fast without shared memory complexity.
            atomicAdd(&out_revenue[suppkey], rev);
            out_present[suppkey] = 1;
        }
    }
}

void q15_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_price,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate,
    int32_t date_lo,
    int32_t date_hi,
    torch::Tensor out_revenue,
    torch::Tensor out_present
) {
    int64_t n = l_suppkey.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    q15_kernel<<<blocks, threads>>>(
        l_suppkey.data_ptr<int64_t>(),
        l_price.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate.data_ptr<int32_t>(),
        date_lo,
        date_hi,
        out_revenue.data_ptr<double>(),
        out_present.data_ptr<int32_t>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", &q15_launcher, "TPC-H Q15 kernel (CUDA)");
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

    Description:
      TPC-H Q15 using a fused CUDA kernel. The lineitem filter and revenue computation
      (price * (1 - discount)) and the aggregation (group by supplier_no) are fused 
      into one pass. Since supplier numbers are contiguous or bounded integers, we can
      directly scatter into a pre-allocated tensor of size MAX(s_suppkey) + 1. 
      cuDF is then used to join the aggregated results with the supplier table.
      
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    # Convert dates to days since epoch
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Tensorize lineitem columns
    l_suppkey = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_price = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate = torch.as_tensor(
        (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32").to_cupy()
    )
    s_suppkey = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)

    # Determine domain size (safe bound for s_suppkey)
    max_supp = int(s_suppkey.max().item())
    
    # Preallocate output arrays for aggregation directly mapped to s_suppkey
    out_revenue = torch.zeros(max_supp + 1, dtype=torch.float64, device=l_suppkey.device)
    out_present = torch.zeros(max_supp + 1, dtype=torch.int32, device=l_suppkey.device)

    # Launch kernel
    _EXT.q15_launcher(
        l_suppkey, l_price, l_discount, l_shipdate,
        shipdate_lo_days, shipdate_hi_days,
        out_revenue, out_present
    )

    # Find the maximum revenue among present suppliers
    present_mask = out_present > 0
    if not present_mask.any():
        empty_res = df_supplier.head(0)[["s_suppkey", "s_name", "s_address", "s_phone"]].copy()
        empty_res["total_revenue"] = cudf.Series(dtype="float64")
        return empty_res
        
    max_rev = out_revenue[present_mask].max()

    # Select suppkeys that matched the max revenue
    sel_mask = present_mask & (out_revenue == max_rev)
    sel_suppkeys = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

    # Move back to cuDF to format final results via an inner join
    sel_suppkeys_sr = cudf.Series(sel_suppkeys)
    sel_rev_sr = cudf.Series(out_revenue[sel_suppkeys])
    
    df_rev = cudf.DataFrame({
        "s_suppkey": sel_suppkeys_sr,
        "total_revenue": sel_rev_sr
    })
    
    # Join with supplier table, fetch only selected properties, and order by suppkey
    res = df_supplier.merge(df_rev, on="s_suppkey", how="inner")
    res = res.sort_values("s_suppkey").reset_index(drop=True)
    
    return res[["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]]
