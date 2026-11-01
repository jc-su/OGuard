import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ inline uint64_t hash_pair(uint64_t p, uint64_t s) {
    uint64_t h = p ^ (s * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 32; h *= 0x85EBCA6B51CE4E43ULL;
    h ^= h >> 33; h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 33;
    return h;
}

__device__ inline uint64_t hash_single(uint64_t s) {
    uint64_t h = s;
    h ^= h >> 32; h *= 0x85EBCA6B51CE4E43ULL;
    h ^= h >> 33; h *= 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 33;
    return h;
}

__global__ void build_ps_hash_kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    int32_t* __restrict__ hash_table,
    int64_t H, int64_t N_ps) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_ps) {
        uint64_t p = ps_partkey[i];
        uint64_t s = ps_suppkey[i];
        uint64_t h = hash_pair(p, s) & (H - 1);
        while (true) {
            int32_t old = atomicCAS(&hash_table[h], -1, (int32_t)i);
            if (old == -1 || old == (int32_t)i) break;
            h = (h + 1) & (H - 1);
        }
    }
}

__global__ void probe_and_agg_kernel(
    const int64_t* __restrict__ l_partkey,
    const int64_t* __restrict__ l_suppkey,
    const double* __restrict__ l_quantity,
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    const int32_t* __restrict__ hash_table,
    double* __restrict__ ps_sum_qty,
    int64_t H, int64_t N_l) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_l) {
        uint64_t p = l_partkey[i];
        uint64_t s = l_suppkey[i];
        uint64_t h = hash_pair(p, s) & (H - 1);
        while (true) {
            int32_t idx = hash_table[h];
            if (idx == -1) break;
            if (ps_partkey[idx] == p && ps_suppkey[idx] == s) {
                atomicAdd(&ps_sum_qty[idx], l_quantity[i]);
                break;
            }
            h = (h + 1) & (H - 1);
        }
    }
}

__global__ void build_supp_hash_kernel(
    const double* __restrict__ ps_sum_qty,
    const int64_t* __restrict__ ps_availqty,
    const int64_t* __restrict__ ps_suppkey,
    unsigned long long int* __restrict__ supp_hash_set,
    int64_t H_s, int64_t N_ps) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_ps) {
        double sum_qty = ps_sum_qty[i];
        if (sum_qty > 0.0 && (double)ps_availqty[i] > 0.5 * sum_qty) {
            uint64_t s = ps_suppkey[i];
            uint64_t h = hash_single(s) & (H_s - 1);
            while (true) {
                unsigned long long int old = atomicCAS(supp_hash_set + h, 
                                                       (unsigned long long int)-1, 
                                                       (unsigned long long int)s);
                if (old == (unsigned long long int)-1 || old == (unsigned long long int)s) break;
                h = (h + 1) & (H_s - 1);
            }
        }
    }
}

__global__ void filter_supplier_kernel(
    const int64_t* __restrict__ s_suppkey,
    const int64_t* __restrict__ s_row_idx,
    const unsigned long long int* __restrict__ supp_hash_set,
    int64_t* __restrict__ out_idx,
    int32_t* __restrict__ out_count,
    int64_t H_s, int64_t N_s) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_s) {
        uint64_t s = s_suppkey[i];
        uint64_t h = hash_single(s) & (H_s - 1);
        bool found = false;
        while (true) {
            unsigned long long int val = supp_hash_set[h];
            if (val == (unsigned long long int)-1) break;
            if (val == (unsigned long long int)s) {
                found = true;
                break;
            }
            h = (h + 1) & (H_s - 1);
        }
        if (found) {
            int32_t pos = atomicAdd(out_count, 1);
            out_idx[pos] = s_row_idx[i];
        }
    }
}

torch::Tensor q20_launcher(
    torch::Tensor l_partkey,
    torch::Tensor l_suppkey,
    torch::Tensor l_quantity,
    torch::Tensor ps_partkey,
    torch::Tensor ps_suppkey,
    torch::Tensor ps_availqty,
    torch::Tensor s_suppkey,
    torch::Tensor s_row_idx
) {
    int64_t N_ps = ps_partkey.numel();
    int64_t N_l = l_partkey.numel();
    int64_t N_s = s_suppkey.numel();

    int64_t H = 1;
    while (H < N_ps * 2) H *= 2;
    if (H < 1024) H = 1024;

    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(ps_partkey.device());
    auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).device(ps_partkey.device());
    auto opt_f64 = torch::TensorOptions().dtype(torch::kFloat64).device(ps_partkey.device());

    auto hash_table = torch::full({H}, -1, opt_i32);
    auto ps_sum_qty = torch::zeros({N_ps}, opt_f64);

    int threads = 256;
    
    int blocks_ps = (N_ps + threads - 1) / threads;
    if (blocks_ps > 0) {
        build_ps_hash_kernel<<<blocks_ps, threads>>>(
            ps_partkey.data_ptr<int64_t>(), ps_suppkey.data_ptr<int64_t>(),
            hash_table.data_ptr<int32_t>(), H, N_ps
        );
    }

    int blocks_l = (N_l + threads - 1) / threads;
    if (blocks_l > 0) {
        probe_and_agg_kernel<<<blocks_l, threads>>>(
            l_partkey.data_ptr<int64_t>(), l_suppkey.data_ptr<int64_t>(), l_quantity.data_ptr<double>(),
            ps_partkey.data_ptr<int64_t>(), ps_suppkey.data_ptr<int64_t>(),
            hash_table.data_ptr<int32_t>(), ps_sum_qty.data_ptr<double>(), H, N_l
        );
    }

    int64_t H_s = 1;
    while (H_s < N_ps * 2) H_s *= 2;
    if (H_s < 1024) H_s = 1024;
    
    auto supp_hash_set = torch::full({H_s}, -1, opt_i64);
    if (blocks_ps > 0) {
        build_supp_hash_kernel<<<blocks_ps, threads>>>(
            ps_sum_qty.data_ptr<double>(), ps_availqty.data_ptr<int64_t>(), ps_suppkey.data_ptr<int64_t>(),
            (unsigned long long int*)supp_hash_set.data_ptr<int64_t>(), H_s, N_ps
        );
    }

    auto out_idx = torch::empty({N_s}, opt_i64);
    auto out_count = torch::zeros({1}, opt_i32);
    
    int blocks_s = (N_s + threads - 1) / threads;
    if (blocks_s > 0) {
        filter_supplier_kernel<<<blocks_s, threads>>>(
            s_suppkey.data_ptr<int64_t>(), s_row_idx.data_ptr<int64_t>(),
            (const unsigned long long int*)supp_hash_set.data_ptr<int64_t>(),
            out_idx.data_ptr<int64_t>(), out_count.data_ptr<int32_t>(), H_s, N_s
        );
    }

    int32_t count = out_count.item<int32_t>();
    return out_idx.slice(0, 0, count);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q20_launcher", &q20_launcher, "TPC-H Q20 Optimized CUDA Kernel");
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
    def to_c(t):
        if not t.is_cuda:
            t = t.cuda()
        return t.contiguous()

    return _EXT.q20_launcher(
        to_c(l_partkey), to_c(l_suppkey), to_c(l_quantity),
        to_c(ps_partkey), to_c(ps_suppkey), to_c(ps_availqty),
        to_c(s_suppkey), to_c(s_row_idx)
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
    supplier_in_nation = supplier_in_nation.reset_index(drop=True)
    s_row_idx_series = cudf.Series(
        cudf.core.column.arange(0, len(supplier_in_nation), dtype="int64")
    ) if False else cudf.Series(range(len(supplier_in_nation))).astype("int64")

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
