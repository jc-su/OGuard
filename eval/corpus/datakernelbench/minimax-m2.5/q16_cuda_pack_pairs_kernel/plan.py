import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <algorithm>
#include <cstdint>

#define BLOCK_SIZE 256

// Kernel to pack group_id and supp_id into a single 64-bit integer
// high 32 bits = group_id, low 32 bits = supp_id
__global__ void pack_pairs_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t* __restrict__ packed,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        packed[idx] = (group_ids[idx] << 32) | (supp_ids[idx] & 0xFFFFFFFFLL);
    }
}

// Kernel for counting unique (group, supp) pairs using atomic add
__global__ void count_unique_atomic_kernel(
    const int64_t* __restrict__ sorted_packed_pairs,
    int64_t* __restrict__ group_counts,
    int64_t num_pairs
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    
    int64_t packed = sorted_packed_pairs[idx];
    int64_t group_id = packed >> 32;
    int64_t supp_id = packed & 0xFFFFFFFFLL;
    
    // Check if this is a new unique pair (different from previous)
    bool is_new_unique = true;
    if (idx > 0) {
        int64_t prev_packed = sorted_packed_pairs[idx - 1];
        int64_t prev_group = prev_packed >> 32;
        int64_t prev_supp = prev_packed & 0xFFFFFFFFLL;
        if (group_id == prev_group && supp_id == prev_supp) {
            is_new_unique = false;
        }
    }
    
    if (is_new_unique) {
        // Use atomicAdd with unsigned long long cast for int64
        atomicAdd((unsigned long long*)&group_counts[group_id], 1ULL);
    }
}

// Launcher for distinct count
// Takes: group_ids (int64), supp_ids (int64)
// Returns: counts per group (int64 tensor)
torch::Tensor count_distinct_launcher(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    int64_t num_groups
) {
    TORCH_CHECK(group_ids.is_cuda(), "group_ids must be on CUDA");
    TORCH_CHECK(supp_ids.is_cuda(), "supp_ids must be on CUDA");
    TORCH_CHECK(group_ids.scalar_type() == torch::kInt64, "group_ids must be int64");
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64, "supp_ids must be int64");
    
    int64_t n = group_ids.numel();
    TORCH_CHECK(n == supp_ids.numel(), "group_ids and supp_ids must have same length");
    
    // Allocate packed pairs tensor
    auto packed = torch::empty({n}, group_ids.options());
    
    // Pack the pairs using kernel
    int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pack_pairs_kernel<<<blocks, BLOCK_SIZE>>>(
        group_ids.data_ptr<int64_t>(),
        supp_ids.data_ptr<int64_t>(),
        packed.data_ptr<int64_t>(),
        n
    );
    cudaDeviceSynchronize();
    
    // Sort the packed pairs using PyTorch's sort
    auto sort_result = torch::sort(packed, 0);
    auto sorted = std::get<0>(sort_result);  // Get the values tensor
    
    // Allocate output counts
    auto counts = torch::zeros({num_groups}, group_ids.options());
    
    // Launch counting kernel using atomic add per unique pair
    blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    count_unique_atomic_kernel<<<blocks, BLOCK_SIZE>>>(
        sorted.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(),
        n
    );
    
    cudaDeviceSynchronize();
    
    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_distinct_launcher", &count_distinct_launcher, 
          "Count distinct suppliers per group using CUDA");
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


def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct ps_suppkey per group using CUDA.
    
    Hot path optimization:
    - Uses CUDA for sort + parallel counting of unique (group, supplier) pairs
    - Packs (group_id, supp_id) into 64-bit integers for efficient sorting
    - Single CUDA kernel after sort to count unique suppliers per group
    - Uses atomicAdd for thread-safe increment per unique (group, supplier) pair
    
    group_ids: int64 tensor in [0, num_groups)
    supp_ids: int64 tensor of ps_suppkey values (already filtered).
    """
    # Ensure tensors are on CUDA and contiguous
    if not group_ids.is_cuda:
        group_ids = group_ids.cuda()
    if not supp_ids.is_cuda:
        supp_ids = supp_ids.cuda()
    
    group_ids = group_ids.contiguous()
    supp_ids = supp_ids.contiguous()
    
    # Call CUDA kernel for distinct count
    counts = _EXT.count_distinct_launcher(
        group_ids,
        supp_ids,
        num_groups
    )
    
    return counts


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
                 
                 Implementation: cuDF for filtering/join prep (string ops, pandas-like),
                 CUDA for the hot path: distinct count aggregation via sort + atomic add.
    
    Params:
        excluded_brand (str): brand to exclude.
        excluded_type_prefix (str): type prefix to exclude (LIKE 'prefix%').
        sizes (tuple[int, ...]): allowed p_size values.
        bad_supplier_comment_pattern (str): SQL LIKE pattern for supplier comments to exclude.
    """
    df_part = tables["part"]
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]

    # --- Filter part table (cuDF handles strings efficiently) ---
    part_filtered = df_part[
        (df_part["p_brand"] != excluded_brand)
        & (~df_part["p_type"].str.startswith(excluded_type_prefix))
        & (df_part["p_size"].isin(list(sizes)))
    ][["p_partkey", "p_brand", "p_type", "p_size"]]

    # --- Bad suppliers (subquery) - cuDF string matching ---
    bad_suppliers = df_supplier[
        df_supplier["s_comment"].str.match(
            bad_supplier_comment_pattern.replace("%", ".*")
        )
    ]["s_suppkey"]

    # --- Filter partsupp: exclude bad suppliers ---
    ps_filtered = df_partsupp[
        ~df_partsupp["ps_suppkey"].isin(bad_suppliers)
    ][["ps_partkey", "ps_suppkey"]]

    # --- Join part ⨝ partsupp (cuDF join) ---
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

    # --- Build group keys (factorize on cuDF side for string columns) ---
    group_df = joined[["p_brand", "p_type", "p_size"]].drop_duplicates().reset_index(drop=True)
    group_df["_gid"] = cudf.Series(range(len(group_df)), dtype="int64")

    # Merge back to get group IDs for each joined row
    joined_with_gid = joined.merge(group_df, on=["p_brand", "p_type", "p_size"])

    num_groups = len(group_df)

    # --- CUDA hot path: distinct count per group ---
    group_ids_t = torch.as_tensor(joined_with_gid["_gid"].to_cupy(), dtype=torch.int64)
    supp_ids_t = torch.as_tensor(joined_with_gid["ps_suppkey"].to_cupy(), dtype=torch.int64)

    # Call CUDA kernel for distinct count
    counts_t = _query_core(group_ids_t, supp_ids_t, num_groups)

    # --- Assemble result (cuDF for final DataFrame) ---
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
