import os
import tempfile
import torch
import cudf
import pandas as pd
from typing import Tuple
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define BLOCK_SIZE 1024

__global__ void fill_hash_buckets_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t n,
    int64_t* __restrict__ keys,  // hash = group_id * 1000003 + supp_id
    int64_t* __restrict__ indices
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t g = group_ids[idx];
        int64_t s = supp_ids[idx];
        keys[idx] = g * 1000003 + s;  // hash
        indices[idx] = idx;
    }
}

__global__ void count_runs_kernel(
    const int64_t* __restrict__ sorted_keys,
    const int64_t* __restrict__ sorted_indices,
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t n,
    int64_t* __restrict__ out_counts
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bool is_new_run = (idx == 0) || (sorted_keys[idx] != sorted_keys[idx-1]);
        if (is_new_run) {
            int64_t orig_idx = sorted_indices[idx];
            int64_t g = group_ids[orig_idx];
            atomicAdd((unsigned long long*)&out_counts[g], 1ULL);
        }
    }
}

torch::Tensor count_distinct_correct_launcher(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    int64_t num_groups
) {
    TORCH_CHECK(group_ids.scalar_type() == torch::kInt64);
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64);
    
    if (!group_ids.is_cuda()) group_ids = group_ids.cuda();
    if (!supp_ids.is_cuda()) supp_ids = supp_ids.cuda();
    group_ids = group_ids.contiguous();
    supp_ids = supp_ids.contiguous();
    
    int64_t n = group_ids.numel();
    TORCH_CHECK(n == supp_ids.numel());
    
    // Allocate temporary buffers
    auto keys = torch::empty({n}, group_ids.options());
    auto indices = torch::empty({n}, group_ids.options());
    auto out = torch::zeros({num_groups}, group_ids.options());
    
    // Fill hash keys and indices
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill_hash_buckets_kernel<<<blocks, BLOCK_SIZE>>>(
        group_ids.data_ptr<int64_t>(),
        supp_ids.data_ptr<int64_t>(),
        n,
        keys.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>()
    );
    
    // Sort pairs by key using cub::DeviceRadixSort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        keys.data_ptr<int64_t>(), keys.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
        n
    );
    
    auto temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    d_temp_storage = temp_storage.data_ptr();
    
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        keys.data_ptr<int64_t>(), keys.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
        n
    );
    
    // Count runs (unique pairs) and atomically add to group counts
    count_runs_kernel<<<blocks, BLOCK_SIZE>>>(
        keys.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),
        group_ids.data_ptr<int64_t>(),
        supp_ids.data_ptr<int64_t>(),
        n,
        out.data_ptr<int64_t>()
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_distinct_correct_launcher", &count_distinct_correct_launcher, "Correct count distinct suppliers per group using sort (CUDA)");
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
    )

_EXT = _load_cuda()

def _launch_count_distinct(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int
) -> torch.Tensor:
    return _EXT.count_distinct_correct_launcher(group_ids, supp_ids, num_groups)

def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct ps_suppkey per group using fused CUDA kernel with radix sort.
    group_ids: int64 tensor in [0, num_groups)
    supp_ids: int64 tensor of ps_suppkey values (already filtered).
    Returns: int64 tensor of length num_groups with distinct counts.
    
    Uses cub::DeviceRadixSort to sort pairs by hash, then counts runs to
    correctly deduplicate across all blocks. This is faster than reference
    torch.unique + scatter_add because sorting is highly optimized.
    """
    return _launch_count_distinct(group_ids, supp_ids, num_groups)


def run_query(
    tables: dict[str, cudf.DataFrame],
    excluded_brand: str = "Brand#45",
    excluded_type_prefix: str = "MEDIUM POLISHED",
    sizes: tuple[int, ...] = (49, 14, 23, 45, 19, 3, 36, 9),
    bad_supplier_comment_pattern: str = "%Customer%Complaints%",
) -> cudf.DataFrame:
    """
    SQL:
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
    Description: TPC-H Q16 — count distinct suppliers per (brand, type, size) with
                 filters on part attributes and excluding suppliers whose comment
                 matches a complaint pattern.
    Params:
        excluded_brand (str): brand to exclude.
        excluded_type_prefix (str): type prefix to exclude (LIKE 'prefix%').
        sizes (tuple[int, ...]): allowed p_size values.
        bad_supplier_comment_pattern (str): SQL LIKE pattern for supplier comments to exclude.
    """
    df_part = tables["part"]
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]

    # --- Filter part table ---
    part_filtered = df_part[
        (df_part["p_brand"] != excluded_brand)
        & (~df_part["p_type"].str.startswith(excluded_type_prefix))
        & (df_part["p_size"].isin(list(sizes)))
    ][["p_partkey", "p_brand", "p_type", "p_size"]]

    # --- Bad suppliers (subquery) ---
    bad_suppliers = df_supplier[
        df_supplier["s_comment"].str.match(
            bad_supplier_comment_pattern.replace("%", ".*")
        )
    ]["s_suppkey"]

    # --- Filter partsupp: exclude bad suppliers ---
    ps_filtered = df_partsupp[
        ~df_partsupp["ps_suppkey"].isin(bad_suppliers)
    ][["ps_partkey", "ps_suppkey"]]

    # --- Join part ⨝ partsupp ---
    joined = part_filtered.merge(
        ps_filtered, left_on="p_partkey", right_on="ps_partkey"
    )

    if len(joined) == 0:
        return cudf.DataFrame({
            "p_brand": cudf.Series([], dtype="object"),
            "p_type": cudf.Series([], dtype="object"),
            "p_size": cudf.Series([], dtype="int32"),
            "supplier_cnt": cudf.Series([], dtype="int64"),
        })

    # --- Build group keys on host/cuDF side (strings can't go to torch) ---
    # Factorize composite (p_brand, p_type, p_size) into integer group ids.
    group_df = joined[["p_brand", "p_type", "p_size"]].drop_duplicates().reset_index(drop=True)
    group_df["_gid"] = cudf.Series(range(len(group_df)), dtype="int64")

    joined_with_gid = joined.merge(group_df, on=["p_brand", "p_type", "p_size"])

    num_groups = len(group_df)

    group_ids_t = torch.as_tensor(joined_with_gid["_gid"].to_cupy(), dtype=torch.int64)
    supp_ids_t = torch.as_tensor(joined_with_gid["ps_suppkey"].to_cupy(), dtype=torch.int64)

    counts_t = _query_core(group_ids_t, supp_ids_t, num_groups)

    # --- Assemble result ---
    result = cudf.DataFrame({
        "p_brand": group_df["p_brand"],
        "p_type": group_df["p_type"],
        "p_size": group_df["p_size"],
        "supplier_cnt": cudf.Series(counts_t, dtype="int64"),
    })

    result = result.sort_values(
        by=["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return result
