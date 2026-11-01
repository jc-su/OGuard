import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q15_kernel(
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int32_t* __restrict__ l_shipdate_days,
    double* __restrict__ revenue_accum,
    int* __restrict__ present_accum,
    int32_t date_lo,
    int32_t date_hi,
    int64_t n_sup,
    int64_t total_rows
) {
    __shared__ double smem[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double total = 0.0;
    int64_t current_key = -1;

    for (int64_t i = idx; i < total_rows; i += blockDim.x * gridDim.x) {
        if (l_shipdate_days[i] >= date_lo && l_shipdate_days[i] < date_hi) {
            double rev = l_extendedprice[i] * (1.0 - l_discount[i]);
            int64_t sk = l_suppkey[i];
            if (sk == current_key) {
                total += rev;
            } else {
                if (current_key != -1) {
                    atomicAdd(revenue_accum + current_key, total);
                }
                current_key = sk;
                total = rev;
            }
            atomicMax(present_accum + sk, 1);
        }
    }

    if (current_key != -1) {
        atomicAdd(revenue_accum + current_key, total);
    }
}

void q15_launcher(
    torch::Tensor l_suppkey,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor l_shipdate_days,
    int32_t date_lo,
    int32_t date_hi,
    torch::Tensor revenue_accum,
    torch::Tensor present_accum
) {
    int blocks = (l_suppkey.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int threads = BLOCK_SIZE;

    q15_kernel<<<blocks, threads>>>(
        l_suppkey.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        l_shipdate_days.data_ptr<int32_t>(),
        revenue_accum.data_ptr<double>(),
        present_accum.data_ptr<int>(),
        date_lo, date_hi,
        revenue_accum.numel(),
        l_suppkey.numel()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q15_launcher", &q15_launcher, "TPC-H Q15 Kernel");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q15_cuda_")
    cu_path = os.path.join(tmpdir, "q15.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q15_cuda",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-maxrregcount=128", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

EXT = _load_cuda()

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
    Description: CUDA-optimized TPC-H Q15 with fused filter/aggregation and final post-processing on GPU
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64).cuda().contiguous()
    l_price_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64).cuda().contiguous()
    l_disc_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64).cuda().contiguous()
    l_shipdate_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    ).cuda().contiguous()

    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)

    n_sup = len(df_supplier)
    revenue_accum = torch.zeros(n_sup, dtype=torch.float64, device="cuda")
    present_accum = torch.zeros(n_sup, dtype=torch.int32, device="cuda")

    EXT.q15_launcher(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_t,
        shipdate_lo_days,
        shipdate_hi_days,
        revenue_accum,
        present_accum
    )

    present = present_accum.bool()
    max_revenue = revenue_accum[present].max()
    best_suppliers = torch.where(present & (revenue_accum == max_revenue))[0]
    best_suppliers_sorted = best_suppliers.sort().values

    # Create selection mask for suppliers
    provider_mask = torch.zeros_like(s_suppkey_t, dtype=torch.bool)
    for sup_id in best_suppliers_sorted:
        provider_mask |= (s_suppkey_t == sup_id)

    provider_indices = provider_mask.nonzero(as_tuple=True)[0]
    provider_info = df_supplier.iloc[provider_indices.cpu()].reset_index(drop=True)

    # Sort results by s_suppkey
    provider_info = provider_info.sort_values("s_suppkey").reset_index(drop=True)

    return cudf.DataFrame({
        "s_suppkey": provider_info["s_suppkey"],
        "s_name": provider_info["s_name"],
        "s_address": provider_info["s_address"],
        "s_phone": provider_info["s_phone"],
        "total_revenue": cudf.Series(revenue_accum[best_suppliers].cpu().cpu()),
    })
