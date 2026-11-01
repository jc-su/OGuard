import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cstdint>

#define BLOCK_SIZE 1024

__global__ void q20_filter_kernel(
    const int64_t* __restrict__ ps_suppkey,
    const int64_t* __restrict__ ps_availqty,
    const double* __restrict__ sum_qty,
    const int64_t* __restrict__ ps_group,
    int64_t n_ps,
    int64_t* __restrict__ qualifying_suppkeys,
    int64_t* __restrict__ n_qualifying
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_ps) {
        double sum_val = sum_qty[ps_group[idx]];
        if ((double)ps_availqty[idx] > 0.5 * sum_val && sum_val > 0.0) {
            int64_t pos = atomicAdd((unsigned long long*)n_qualifying, 1ULL);
            qualifying_suppkeys[pos] = ps_suppkey[idx];
        }
    }
}

__global__ void q20_isin_kernel(
    const int64_t* __restrict__ s_suppkey,
    const int64_t* __restrict__ s_row_idx,
    const int64_t* __restrict__ qual_suppkeys,
    int64_t n_s,
    int64_t n_qual,
    int64_t* __restrict__ out_indices,
    int64_t* __restrict__ n_out
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_s) {
        int64_t sk = s_suppkey[idx];
        int64_t found = 0;
        for (int64_t i = 0; i < n_qual; i++) {
            if (qual_suppkeys[i] == sk) {
                found = 1;
                break;
            }
        }
        if (found) {
            int64_t pos = atomicAdd((unsigned long long*)n_out, 1ULL);
            out_indices[pos] = s_row_idx[idx];
        }
    }
}

void q20_filter_launcher(
    torch::Tensor ps_suppkey,
    torch::Tensor ps_availqty,
    torch::Tensor sum_qty,
    torch::Tensor ps_group,
    torch::Tensor qualifying_suppkeys,
    torch::Tensor n_qualifying
) {
    int64_t n_ps = ps_suppkey.numel();
    int blocks = (n_ps + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q20_filter_kernel<<<blocks, BLOCK_SIZE>>>(
        ps_suppkey.data_ptr<int64_t>(),
        ps_availqty.data_ptr<int64_t>(),
        sum_qty.data_ptr<double>(),
        ps_group.data_ptr<int64_t>(),
        n_ps,
        qualifying_suppkeys.data_ptr<int64_t>(),
        n_qualifying.data_ptr<int64_t>()
    );
}

int64_t q20_isin_launcher(
    torch::Tensor s_suppkey,
    torch::Tensor s_row_idx,
    torch::Tensor qual_suppkeys,
    torch::Tensor out_indices
) {
    int64_t n_s = s_suppkey.numel();
    int64_t n_qual = qual_suppkeys.numel();
    int blocks = (n_s + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    auto n_out = torch::zeros({1}, torch::kInt64).cuda();
    
    q20_isin_kernel<<<blocks, BLOCK_SIZE>>>(
        s_suppkey.data_ptr<int64_t>(),
        s_row_idx.data_ptr<int64_t>(),
        qual_suppkeys.data_ptr<int64_t>(),
        n_s, n_qual,
        out_indices.data_ptr<int64_t>(),
        n_out.data_ptr<int64_t>()
    );
    
    return n_out.item<int64_t>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q20_filter_launcher", &q20_filter_launcher, "Filter partsupp using sum_qty comparison");
    m.def("q20_isin_launcher", &q20_isin_launcher, "Isin kernel for supplier key matching");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q20_cuda_")
    cu_path = os.path.join(tmpdir, "_q20_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q20_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
    s_suppkey: torch.Tensor,
    s_row_idx: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA-accelerated version:
    - Build group keys using torch.unique (GPU-optimized)
    - Use scatter_add_ for aggregation (GPU-optimized)
    - Use custom CUDA kernel for filtering and isin matching
    """
    # Ensure all tensors are on CUDA
    l_partkey = l_partkey.cuda()
    l_suppkey = l_suppkey.cuda()
    l_quantity = l_quantity.cuda()
    ps_partkey = ps_partkey.cuda()
    ps_suppkey = ps_suppkey.cuda()
    ps_availqty = ps_availqty.cuda()
    s_suppkey = s_suppkey.cuda()
    s_row_idx = s_row_idx.cuda()
    
    # Build composite keys
    ps_pair = torch.stack([ps_partkey, ps_suppkey], dim=1)
    li_pair = torch.stack([l_partkey, l_suppkey], dim=1)
    
    all_pairs = torch.cat([ps_pair, li_pair], dim=0)
    unique_pairs, inverse = torch.unique(all_pairs, dim=0, return_inverse=True)
    
    Nps = ps_pair.shape[0]
    ps_group = inverse[:Nps]
    li_group = inverse[Nps:]
    
    num_groups = unique_pairs.shape[0]
    sum_qty = torch.zeros(num_groups, dtype=l_quantity.dtype, device=l_quantity.device)
    sum_qty.scatter_add_(0, li_group, l_quantity)
    
    # Use CUDA kernel to filter partsupp
    ps_suppkey_cont = ps_suppkey.contiguous()
    ps_availqty_cont = ps_availqty.contiguous()
    sum_qty_cont = sum_qty.contiguous()
    ps_group_cont = ps_group.contiguous()
    
    # First pass: count qualifying
    n_ps = ps_suppkey.shape[0]
    count_tensor = torch.zeros(1, dtype=torch.int64, device=ps_suppkey.device)
    
    # Allocate max possible output
    qualifying_suppkeys = torch.zeros(n_ps, dtype=torch.int64, device=ps_suppkey.device)
    
    _EXT.q20_filter_launcher(
        ps_suppkey_cont, ps_availqty_cont, sum_qty_cont, ps_group_cont,
        qualifying_suppkeys, count_tensor
    )
    
    n_qual = count_tensor.item()
    if n_qual == 0:
        return torch.empty(0, dtype=torch.int64, device=l_partkey.device)
    
    # Get unique qualifying suppkeys
    qual_suppkeys = qualifying_suppkeys[:n_qual]
    unique_qual = torch.unique(qual_suppkeys).contiguous()
    
    # Use CUDA kernel for isin matching
    s_suppkey_cont = s_suppkey.contiguous()
    s_row_idx_cont = s_row_idx.contiguous()
    
    out_indices = torch.zeros(s_suppkey.shape[0], dtype=torch.int64, device=s_suppkey.device)
    n_out = _EXT.q20_isin_launcher(
        s_suppkey_cont, s_row_idx_cont, unique_qual, out_indices
    )
    
    return out_indices[:n_out]


def run_query(
    tables: dict[str, cudf.DataFrame],
    name_prefix: str = "forest",
    nation_name: str = "CANADA",
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT s_name, s_address FROM supplier, nation WHERE s_suppkey IN (
            SELECT ps_suppkey FROM partsupp WHERE ps_partkey IN (
                SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
            AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem
                               WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
                               AND l_shipdate >= DATE '1994-01-01'
                               AND l_shipdate <  DATE '1995-01-01'))
          AND s_nationkey = n_nationkey AND n_name = 'CANADA'
          ORDER BY s_name;
    Description: TPC-H Q20 — finds suppliers in a specific nation who have excess inventory
                 of parts whose name starts with a given prefix during a date range.
    Params:
        name_prefix (str): prefix filter for p_name (LIKE '<prefix>%').
        nation_name (str): nation filter for supplier's nation.
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_partsupp = tables["partsupp"]
    df_part = tables["part"]
    df_lineitem = tables["lineitem"]

    # Convert date parameters to epoch days (int32)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # --- Filter part by p_name LIKE 'prefix%' ---
    part_mask = df_part["p_name"].str.startswith(name_prefix)
    filtered_part = df_part[part_mask][["p_partkey"]]

    # --- Filter partsupp by ps_partkey IN filtered_part ---
    filtered_partsupp = df_partsupp.merge(
        filtered_part, left_on="ps_partkey", right_on="p_partkey", how="inner"
    )[["ps_partkey", "ps_suppkey", "ps_availqty"]]

    # --- Filter lineitem by shipdate range ---
    l_shipdate_days = (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")
    li_mask = (l_shipdate_days >= shipdate_lo_days) & (l_shipdate_days < shipdate_hi_days)
    filtered_lineitem = df_lineitem[li_mask][["l_partkey", "l_suppkey", "l_quantity"]]

    # --- Filter supplier by nation = CANADA ---
    nation_filtered = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supplier_in_nation = df_supplier.merge(
        nation_filtered, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    ).reset_index(drop=True)

    # Add a row index column so we can recover s_name / s_address by tensor index
    s_row_idx_series = cudf.Series(range(len(supplier_in_nation))).astype("int64")

    # Convert tensors
    l_partkey_t = torch.as_tensor(filtered_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_suppkey_t = torch.as_tensor(filtered_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(filtered_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    ps_partkey_t = torch.as_tensor(filtered_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(filtered_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    ps_availqty_t = torch.as_tensor(filtered_partsupp["ps_availqty"].to_cupy(), dtype=torch.int64)

    s_suppkey_t = torch.as_tensor(supplier_in_nation["s_suppkey"].to_cupy(), dtype=torch.int64)
    s_row_idx_t = torch.as_tensor(s_row_idx_series.to_cupy(), dtype=torch.int64)

    selected_idx_t = _query_core(
        l_partkey_t, l_suppkey_t, l_quantity_t,
        ps_partkey_t, ps_suppkey_t, ps_availqty_t,
        s_suppkey_t, s_row_idx_t,
    )

    # Materialize result via gather on cuDF side
    selected_idx_cudf = cudf.Series(selected_idx_t)
    result = supplier_in_nation.iloc[selected_idx_cudf.to_pandas().tolist()][["s_name", "s_address"]]
    result = result.sort_values("s_name").reset_index(drop=True)
    return result
