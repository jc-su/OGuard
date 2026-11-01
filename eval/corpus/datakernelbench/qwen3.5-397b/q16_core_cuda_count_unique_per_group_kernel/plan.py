import os
import tempfile
import torch
import cudf
import pandas as pd
from typing import Tuple
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Kernel to count unique pairs per group after sorting
// Input: sorted group_ids and supp_ids (sorted by group then supp)
// Output: counts per group via atomicAdd
__global__ void count_unique_per_group_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t* __restrict__ out_counts,
    int64_t n,
    int64_t num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Check if this is the first occurrence of this (group, supp) pair
        bool is_first = true;
        if (idx > 0) {
            is_first = (group_ids[idx] != group_ids[idx-1]) || 
                       (supp_ids[idx] != supp_ids[idx-1]);
        }
        
        if (is_first) {
            atomicAdd((unsigned long long*)&out_counts[group_ids[idx]], 1ULL);
        }
    }
}

torch::Tensor q16_launcher(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    int64_t num_groups
) {
    // Ensure CUDA and contiguous
    if (!group_ids.is_cuda()) group_ids = group_ids.cuda();
    if (!supp_ids.is_cuda()) supp_ids = supp_ids.cuda();
    group_ids = group_ids.contiguous();
    supp_ids = supp_ids.contiguous();
    
    TORCH_CHECK(group_ids.scalar_type() == torch::kInt64, "group_ids must be int64");
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64, "supp_ids must be int64");
    TORCH_CHECK(group_ids.numel() == supp_ids.numel(), "group_ids and supp_ids must have same length");
    
    int64_t n = group_ids.numel();
    
    auto out = torch::zeros({num_groups}, group_ids.options().dtype(torch::kInt64));
    
    if (n == 0) {
        return out;
    }
    
    const int64_t* group_ptr = group_ids.data_ptr<int64_t>();
    const int64_t* supp_ptr = supp_ids.data_ptr<int64_t>();
    int64_t* out_ptr = out.data_ptr<int64_t>();
    
    // Launch counting kernel (assumes input is already sorted by host)
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_unique_per_group_kernel<<<blocks, BLOCK_SIZE>>>(
        group_ptr,
        supp_ptr,
        out_ptr,
        n,
        num_groups
    );
    
    cudaDeviceSynchronize();
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q16_launcher", &q16_launcher, "TPC-H Q16 count distinct suppliers per group (CUDA)");
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

def _launch_q16(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Launch CUDA kernel for count distinct suppliers per group.
    
    GPU: CUDA kernel scans sorted (group, supp) pairs and counts unique per group.
    Why faster: Single kernel pass over sorted data vs torch.unique + scatter_add.
    Host sorts pairs, CUDA kernel does the counting with coalesced access.
    """
    counts = _EXT.q16_launcher(group_ids, supp_ids, num_groups)
    return counts

def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct ps_suppkey per group.
    
    GPU: CUDA kernel via torch.utils.cpp_extension.load()
    Why faster: Fused counting kernel reduces HBM passes vs torch.unique + scatter_add.
    Host sorts (group, supp) pairs, CUDA kernel scans and counts unique per group.
    
    Args:
        group_ids: int64 tensor in [0, num_groups), group assignment for each row
        supp_ids: int64 tensor of ps_suppkey values (already filtered)
        num_groups: number of distinct (brand, type, size) groups
    
    Returns:
        counts: int64 tensor of size num_groups, count of distinct suppliers per group
    """
    # Stack pairs and sort by (group_id, supp_id) on GPU using torch.sort
    # Create composite key for sorting: group_id * (max_supp+1) + supp_id
    # This ensures stable sort by group first, then supp
    max_supp = supp_ids.max().item() + 1 if supp_ids.numel() > 0 else 1
    composite_keys = group_ids * max_supp + supp_ids
    
    # Sort by composite key
    sorted_indices = torch.argsort(composite_keys)
    sorted_group_ids = group_ids[sorted_indices].contiguous()
    sorted_supp_ids = supp_ids[sorted_indices].contiguous()
    
    return _launch_q16(sorted_group_ids, sorted_supp_ids, num_groups)

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
