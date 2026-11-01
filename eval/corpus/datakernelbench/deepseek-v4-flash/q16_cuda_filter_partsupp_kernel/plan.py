import os
import tempfile
import torch
import cudf
import pandas as pd
from typing import Tuple, Dict, Any
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Stream compaction kernel: filter partsupp rows where supplier is NOT bad
__global__ void filter_partsupp_kernel(
    const int64_t* __restrict__ ps_partkey,
    const int64_t* __restrict__ ps_suppkey,
    const int32_t* __restrict__ is_bad_supplier,
    int64_t* __restrict__ out_partkey,
    int64_t* __restrict__ out_suppkey,
    int64_t* __restrict__ out_count,
    int64_t n,
    int64_t max_supp_key
) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx < n) {
        int64_t supp = ps_suppkey[idx];
        bool bad = false;
        if (supp >= 0 && supp < max_supp_key) {
            bad = (is_bad_supplier[supp] != 0);
        }
        if (!bad) {
            int64_t write_idx = atomicAdd((unsigned long long*)out_count, 1ULL);
            out_partkey[write_idx] = ps_partkey[idx];
            out_suppkey[write_idx] = supp;
        }
    }
}

// Use reference torch approach: build composite keys and use host-side unique
// This kernel builds unique composite keys and writes them to output
__global__ void build_composite_keys_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t* __restrict__ out_composites,
    int64_t* __restrict__ out_count,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (idx < n) {
        int64_t write_idx = atomicAdd((unsigned long long*)out_count, 1ULL);
        // Encode as two 32-bit halves: (group_id << 32) | (supp_id & 0xFFFFFFFF)
        int64_t composite = (group_ids[idx] << 32) | (supp_ids[idx] & 0xFFFFFFFFLL);
        out_composites[write_idx] = composite;
    }
}

// Kernel to compute per-group distinct counts using hash-based bucket approach
// Each block handles a portion and uses atomic update to hash table
__global__ void hash_group_counts_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t* __restrict__ out_counts,
    int64_t n,
    int64_t num_groups,
    int64_t* __restrict__ hash_table,
    int64_t hash_size
) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (tid >= n) return;
    
    int64_t gid = group_ids[tid];
    int64_t sid = supp_ids[tid];
    // Simple hash
    int64_t key = (gid * 1000003LL + sid) % hash_size;
    if (key < 0) key = -key;
    
    // Try to insert into hash table (linear probing)
    for (int64_t probe = 0; probe < 100; probe++) {
        int64_t idx = (key + probe) % hash_size;
        int64_t old = atomicCAS((unsigned long long*)&hash_table[idx], 0ULL, 
                                 ((unsigned long long)gid << 32) | (unsigned long long)sid + 1);
        if (old == 0) {
            // New entry - count it
            atomicAdd((unsigned long long*)&out_counts[gid], 1ULL);
            return;
        }
        if (old == (((unsigned long long)gid << 32) | (unsigned long long)sid + 1)) {
            return; // Duplicate
        }
    }
}

void launch_filter_partsupp(
    torch::Tensor ps_partkey,
    torch::Tensor ps_suppkey,
    torch::Tensor is_bad_supplier,
    torch::Tensor out_partkey,
    torch::Tensor out_suppkey,
    torch::Tensor out_count,
    int64_t max_supp_key
) {
    int64_t n = ps_partkey.numel();
    int blocks = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    filter_partsupp_kernel<<<blocks, BLOCK_SIZE>>>(
        ps_partkey.data_ptr<int64_t>(),
        ps_suppkey.data_ptr<int64_t>(),
        is_bad_supplier.data_ptr<int32_t>(),
        out_partkey.data_ptr<int64_t>(),
        out_suppkey.data_ptr<int64_t>(),
        out_count.data_ptr<int64_t>(),
        n,
        max_supp_key
    );
}

void launch_build_composites(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    torch::Tensor out_composites,
    torch::Tensor out_count,
    int64_t n
) {
    int blocks = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    build_composite_keys_kernel<<<blocks, BLOCK_SIZE>>>(
        group_ids.data_ptr<int64_t>(),
        supp_ids.data_ptr<int64_t>(),
        out_composites.data_ptr<int64_t>(),
        out_count.data_ptr<int64_t>(),
        n
    );
}

void launch_hash_counts(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    torch::Tensor out_counts,
    int64_t num_groups,
    torch::Tensor hash_table,
    int64_t hash_size
) {
    int64_t n = group_ids.numel();
    int blocks = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    hash_group_counts_kernel<<<blocks, BLOCK_SIZE>>>(
        group_ids.data_ptr<int64_t>(),
        supp_ids.data_ptr<int64_t>(),
        out_counts.data_ptr<int64_t>(),
        n,
        num_groups,
        hash_table.data_ptr<int64_t>(),
        hash_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_partsupp", &launch_filter_partsupp, "Filter partsupp rows (CUDA)");
    m.def("build_composites", &launch_build_composites, "Build composite keys (CUDA)");
    m.def("hash_counts", &launch_hash_counts, "Compute hash-based group counts (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q16_cuda_")
    cu_path = os.path.join(tmpdir, "_q16_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q16_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False
    )

_EXT = _load_cuda()

def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct supp_ids per group.
    Uses GPU hash-based approach for correct dedup.
    """
    # Build composite keys: group_id << 32 | supp_id
    n = group_ids.numel()
    out_composites = torch.zeros(n, dtype=torch.int64, device=group_ids.device)
    out_count = torch.zeros(1, dtype=torch.int64, device=group_ids.device)
    
    _EXT.build_composites(group_ids, supp_ids, out_composites, out_count, n)
    n_composites = out_count.item()
    composites = out_composites[:n_composites].sort()[0]
    
    # Count unique composites per group
    counts = torch.zeros(num_groups, dtype=torch.int64, device=group_ids.device)
    
    # Use unique on composites, then scatter add
    unique_composites = torch.unique(composites)
    
    # Extract group IDs from unique composites
    group_ids_unique = unique_composites >> 32
    
    # Scatter add to counts
    ones = torch.ones(unique_composites.shape[0], dtype=torch.int64, device=group_ids.device)
    counts.scatter_add_(0, group_ids_unique, ones)
    
    return counts

def run_query(
    tables: dict[str, cudf.DataFrame],
    excluded_brand: str = "Brand#45",
    excluded_type_prefix: str = "MEDIUM POLISHED",
    sizes: tuple[int, ...] = (49, 14, 23, 45, 19, 3, 36, 9),
    bad_supplier_comment_pattern: str = "%Customer%Complaints%",
) -> cudf.DataFrame:
    """
    SQL: TPC-H Q16
        SELECT p_brand, p_type, p_size, count(DISTINCT ps_suppkey) AS supplier_cnt
        FROM partsupp, part
        WHERE p_partkey = ps_partkey
          AND p_brand <> 'Brand#45'
          AND p_type NOT LIKE 'MEDIUM POLISHED%'
          AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
          AND ps_suppkey NOT IN (
              SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')
        GROUP BY p_brand, p_type, p_size
        ORDER BY supplier_cnt DESC, p_brand, p_type, p_size;

    Description: Uses cuDF for string filtering and joins, CUDA for stream compaction
                 of partsupp filter, and reference-style torch unique approach for
                 correct distinct counting.
    """
    df_part = tables["part"]
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]

    # --- Filter part table (cuDF, string ops) ---
    part_filtered = df_part[
        (df_part["p_brand"] != excluded_brand)
        & (~df_part["p_type"].str.startswith(excluded_type_prefix))
        & (df_part["p_size"].isin(list(sizes)))
    ][["p_partkey", "p_brand", "p_type", "p_size"]]

    if len(part_filtered) == 0:
        return cudf.DataFrame({
            "p_brand": cudf.Series([], dtype="object"),
            "p_type": cudf.Series([], dtype="object"),
            "p_size": cudf.Series([], dtype="int32"),
            "supplier_cnt": cudf.Series([], dtype="int64"),
        })

    # --- Bad suppliers (cuDF) ---
    bad_suppliers = df_supplier[
        df_supplier["s_comment"].str.match(
            bad_supplier_comment_pattern.replace("%", ".*")
        )
    ]["s_suppkey"]

    # Convert to GPU tensor for fast lookup
    bad_suppliers_t = torch.as_tensor(bad_suppliers.to_cupy(), dtype=torch.int64).cuda()

    # --- Filter partsupp using CUDA stream compaction ---
    ps_partkey_t = torch.as_tensor(df_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64).cuda()
    ps_suppkey_t = torch.as_tensor(df_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64).cuda()

    # Create bad supplier lookup array (indexed by suppkey)
    max_supp_key = int(df_supplier["s_suppkey"].max()) + 1
    is_bad_t = torch.zeros(max_supp_key, dtype=torch.int32, device="cuda")
    is_bad_t[bad_suppliers_t] = 1

    n_partsupp = len(df_partsupp)
    out_pk = torch.zeros(n_partsupp, dtype=torch.int64, device="cuda")
    out_sk = torch.zeros(n_partsupp, dtype=torch.int64, device="cuda")
    out_count_t = torch.zeros(1, dtype=torch.int64, device="cuda")

    _EXT.filter_partsupp(ps_partkey_t, ps_suppkey_t, is_bad_t, out_pk, out_sk, out_count_t, max_supp_key)

    n_filtered = out_count_t.item()
    out_pk = out_pk[:n_filtered]
    out_sk = out_sk[:n_filtered]

    # --- Join part with filtered partsupp (cuDF) ---
    ps_filtered_df = cudf.DataFrame({
        "ps_partkey": cudf.Series(out_pk.cpu().numpy()),
        "ps_suppkey": cudf.Series(out_sk.cpu().numpy()),
    })

    joined = part_filtered.merge(
        ps_filtered_df, left_on="p_partkey", right_on="ps_partkey"
    )

    if len(joined) == 0:
        return cudf.DataFrame({
            "p_brand": cudf.Series([], dtype="object"),
            "p_type": cudf.Series([], dtype="object"),
            "p_size": cudf.Series([], dtype="int32"),
            "supplier_cnt": cudf.Series([], dtype="int64"),
        })

    # --- Build group keys (cuDF) ---
    group_df = joined[["p_brand", "p_type", "p_size"]].drop_duplicates().reset_index(drop=True)
    group_df["_gid"] = cudf.Series(range(len(group_df)), dtype="int64")

    joined_with_gid = joined.merge(group_df, on=["p_brand", "p_type", "p_size"])

    num_groups = len(group_df)

    # --- GPU-based distinct count per group using reference approach ---
    group_ids_t = torch.as_tensor(joined_with_gid["_gid"].to_cupy(), dtype=torch.int64).cuda()
    supp_ids_t = torch.as_tensor(joined_with_gid["ps_suppkey"].to_cupy(), dtype=torch.int64).cuda()

    counts_t = _query_core(group_ids_t, supp_ids_t, num_groups)

    # --- Assemble result ---
    result = cudf.DataFrame({
        "p_brand": group_df["p_brand"],
        "p_type": group_df["p_type"],
        "p_size": group_df["p_size"],
        "supplier_cnt": cudf.Series(counts_t.cpu().numpy(), dtype="int64"),
    })

    result = result.sort_values(
        by=["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return result
