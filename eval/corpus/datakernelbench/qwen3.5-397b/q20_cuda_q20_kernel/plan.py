import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Device function to compare two keys (partkey, suppkey)
__device__ __forceinline__ bool key_less(int64_t p1, int64_t s1, int64_t p2, int64_t s2) {
    if (p1 != p2) return p1 < p2;
    return s1 < s2;
}

// Lower bound: first index where key >= target
__device__ int64_t lower_bound(
    const int64_t* p_keys, const int64_t* s_keys, int64_t n,
    int64_t target_p, int64_t target_s
) {
    int64_t low = 0, high = n;
    while (low < high) {
        int64_t mid = low + (high - low) / 2;
        if (key_less(p_keys[mid], s_keys[mid], target_p, target_s)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Upper bound: first index where key > target
__device__ int64_t upper_bound(
    const int64_t* p_keys, const int64_t* s_keys, int64_t n,
    int64_t target_p, int64_t target_s
) {
    int64_t low = 0, high = n;
    while (low < high) {
        int64_t mid = low + (high - low) / 2;
        if (key_less(target_p, target_s, p_keys[mid], s_keys[mid])) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

__global__ void q20_kernel(
    const int64_t* ps_partkey,
    const int64_t* ps_suppkey,
    const double* ps_availqty,
    int64_t n_ps,
    const int64_t* li_partkey,
    const int64_t* li_suppkey,
    const double* li_quantity,
    int64_t n_li,
    int64_t* out_suppkeys,
    int* out_count
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ps) return;

    int64_t p_key = ps_partkey[idx];
    int64_t s_key = ps_suppkey[idx];
    double avail = ps_availqty[idx];

    // Find range in lineitem using binary search
    // Both arrays are sorted by (partkey, suppkey)
    int64_t start = lower_bound(li_partkey, li_suppkey, n_li, p_key, s_key);
    int64_t end = upper_bound(li_partkey, li_suppkey, n_li, p_key, s_key);

    double sum_qty = 0.0;
    // Sum quantities in range
    for (int64_t i = start; i < end; ++i) {
        sum_qty += li_quantity[i];
    }

    // Check condition: sum_qty > 0 AND avail > 0.5 * sum_qty
    // Reference requires sum_qty > 0 (there must be lineitem rows in date range)
    if (sum_qty > 0.0 && avail > 0.5 * sum_qty) {
        int pos = atomicAdd(out_count, 1);
        out_suppkeys[pos] = s_key;
    }
}

torch::Tensor q20_launcher(
    torch::Tensor ps_partkey,
    torch::Tensor ps_suppkey,
    torch::Tensor ps_availqty,
    torch::Tensor li_partkey,
    torch::Tensor li_suppkey,
    torch::Tensor li_quantity
) {
    // Ensure CUDA and contiguous
    auto opt = ps_partkey.options();
    if (!ps_partkey.is_cuda()) ps_partkey = ps_partkey.cuda();
    if (!ps_suppkey.is_cuda()) ps_suppkey = ps_suppkey.cuda();
    if (!ps_availqty.is_cuda()) ps_availqty = ps_availqty.cuda();
    if (!li_partkey.is_cuda()) li_partkey = li_partkey.cuda();
    if (!li_suppkey.is_cuda()) li_suppkey = li_suppkey.cuda();
    if (!li_quantity.is_cuda()) li_quantity = li_quantity.cuda();

    ps_partkey = ps_partkey.contiguous();
    ps_suppkey = ps_suppkey.contiguous();
    ps_availqty = ps_availqty.contiguous();
    li_partkey = li_partkey.contiguous();
    li_suppkey = li_suppkey.contiguous();
    li_quantity = li_quantity.contiguous();

    int64_t n_ps = ps_partkey.numel();
    int64_t n_li = li_partkey.numel();

    // Output buffer: max size n_ps (all qualify)
    auto out_opts = opt.dtype(torch::kInt64);
    torch::Tensor out_suppkeys = torch::empty({n_ps}, out_opts);
    torch::Tensor out_count = torch::zeros({1}, opt.dtype(torch::kInt32).device(opt.device()));

    int blocks = (n_ps + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q20_kernel<<<blocks, BLOCK_SIZE>>>(
        ps_partkey.data_ptr<int64_t>(),
        ps_suppkey.data_ptr<int64_t>(),
        ps_availqty.data_ptr<double>(),
        n_ps,
        li_partkey.data_ptr<int64_t>(),
        li_suppkey.data_ptr<int64_t>(),
        li_quantity.data_ptr<double>(),
        n_li,
        out_suppkeys.data_ptr<int64_t>(),
        out_count.data_ptr<int32_t>()
    );

    // Sync and slice
    cudaDeviceSynchronize();
    int count = out_count.item<int32_t>();
    return out_suppkeys.slice(0, 0, count).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q20_launcher", &q20_launcher, "TPC-H Q20 CUDA Core");
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
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
    li_partkey: torch.Tensor,
    li_suppkey: torch.Tensor,
    li_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Runs the CUDA kernel to find qualifying supplier keys.
    Inputs must be sorted by (partkey, suppkey).
    Replaces reference torch.unique + scatter_add with a sorted merge-join style aggregation.
    """
    return _EXT.q20_launcher(
        ps_partkey, ps_suppkey, ps_availqty,
        li_partkey, li_suppkey, li_quantity
    )

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
                 Optimized with CUDA for the aggregation join between partsupp and lineitem.
                 Uses cuDF for filtering and sorting, CUDA for the hot path aggregation.
                 Avoids large temporary tensors from torch.unique by using sorted binary search.
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

    # Add row index for final gather
    supplier_in_nation = supplier_in_nation.reset_index(drop=True)
    s_row_idx_series = cudf.Series(range(len(supplier_in_nation))).astype("int64")

    # --- Sort for CUDA Merge-Join ---
    # Sort partsupp by (ps_partkey, ps_suppkey)
    filtered_partsupp = filtered_partsupp.sort_values(["ps_partkey", "ps_suppkey"]).reset_index(drop=True)
    # Sort lineitem by (l_partkey, l_suppkey)
    filtered_lineitem = filtered_lineitem.sort_values(["l_partkey", "l_suppkey"]).reset_index(drop=True)

    # Convert to tensors
    ps_partkey_t = torch.as_tensor(filtered_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(filtered_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    ps_availqty_t = torch.as_tensor(filtered_partsupp["ps_availqty"].to_cupy(), dtype=torch.float64)

    li_partkey_t = torch.as_tensor(filtered_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    li_suppkey_t = torch.as_tensor(filtered_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    li_quantity_t = torch.as_tensor(filtered_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    # Run CUDA kernel
    qualifying_suppkeys_t = _query_core(
        ps_partkey_t, ps_suppkey_t, ps_availqty_t,
        li_partkey_t, li_suppkey_t, li_quantity_t,
    )

    # Unique qualifying suppkeys
    if qualifying_suppkeys_t.numel() == 0:
        return supplier_in_nation.iloc[[]][["s_name", "s_address"]].sort_values("s_name").reset_index(drop=True)
    
    unique_qual_supp = torch.unique(qualifying_suppkeys_t)

    # Filter supplier
    s_suppkey_t = torch.as_tensor(supplier_in_nation["s_suppkey"].to_cupy(), dtype=torch.int64)
    s_row_idx_t = torch.as_tensor(s_row_idx_series.to_cupy(), dtype=torch.int64)

    supp_mask = torch.isin(s_suppkey_t, unique_qual_supp)
    selected_idx_t = s_row_idx_t[supp_mask]

    # Materialize result
    if selected_idx_t.numel() == 0:
        return supplier_in_nation.iloc[[]][["s_name", "s_address"]].sort_values("s_name").reset_index(drop=True)

    selected_idx_cudf = cudf.Series(selected_idx_t.cpu().numpy())
    result = supplier_in_nation.iloc[selected_idx_cudf.to_pandas().tolist()][["s_name", "s_address"]]
    result = result.sort_values("s_name").reset_index(drop=True)
    return result
