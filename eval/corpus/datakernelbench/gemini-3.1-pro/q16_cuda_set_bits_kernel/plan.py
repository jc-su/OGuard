import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 256

__global__ void set_bits_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    uint32_t* __restrict__ bitmap,
    int64_t n_elements,
    int32_t num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t g = group_ids[idx];
        int64_t s = supp_ids[idx];
        
        int32_t word_idx = s / 32;
        int32_t bit_idx = s % 32;
        
        // Coalesced access layout: bitmap[word_idx * num_groups + g]
        int64_t offset = (int64_t)word_idx * num_groups + g;
        atomicOr(&bitmap[offset], 1u << bit_idx);
    }
}

__global__ void count_bits_kernel(
    const uint32_t* __restrict__ bitmap,
    int64_t* __restrict__ counts,
    int32_t num_groups,
    int32_t num_words
) {
    int32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < num_groups) {
        int64_t count = 0;
        for (int32_t w = 0; w < num_words; ++w) {
            int64_t offset = (int64_t)w * num_groups + g;
            count += __popc(bitmap[offset]);
        }
        counts[g] = count;
    }
}

torch::Tensor q16_count_distinct_launcher(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    int64_t num_groups,
    int64_t max_supp_id
) {
    TORCH_CHECK(group_ids.is_cuda(), "group_ids must be a CUDA tensor");
    TORCH_CHECK(supp_ids.is_cuda(), "supp_ids must be a CUDA tensor");
    group_ids = group_ids.contiguous();
    supp_ids = supp_ids.contiguous();
    
    int64_t n = group_ids.numel();
    int32_t num_words = (max_supp_id + 32) / 32;
    
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(group_ids.device());
    // Use int32 under the hood but we cast to uint32_t in the kernel
    auto bitmap = torch::zeros({num_words, num_groups}, options);
    
    int blocks_set = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_set > 0) {
        set_bits_kernel<<<blocks_set, BLOCK_SIZE>>>(
            group_ids.data_ptr<int64_t>(),
            supp_ids.data_ptr<int64_t>(),
            reinterpret_cast<uint32_t*>(bitmap.data_ptr<int32_t>()),
            n,
            num_groups
        );
    }
    
    auto counts = torch::zeros({num_groups}, torch::TensorOptions().dtype(torch::kInt64).device(group_ids.device()));
    int blocks_count = (num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_count > 0) {
        count_bits_kernel<<<blocks_count, BLOCK_SIZE>>>(
            reinterpret_cast<uint32_t*>(bitmap.data_ptr<int32_t>()),
            counts.data_ptr<int64_t>(),
            num_groups,
            num_words
        );
    }
    
    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q16_count_distinct_launcher", &q16_count_distinct_launcher, "TPC-H Q16 distinct supplier count");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q16_cuda_")
    cu_path = os.path.join(tmpdir, "kernels.cu")
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

_EXT = None

def _get_ext():
    global _EXT
    if _EXT is None:
        _EXT = _load_cuda()
    return _EXT

def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct ps_suppkey per group using a custom CUDA kernel.
    The kernel builds a dense bitmask per group and counts populated bits.
    This avoids sorting or large host-side factorizations.
    
    group_ids: int64 tensor in [0, num_groups)
    supp_ids: int64 tensor of ps_suppkey values (already filtered).
    """
    if len(group_ids) == 0 or num_groups == 0:
        return torch.zeros(num_groups, dtype=torch.int64, device=group_ids.device)
    max_supp = int(supp_ids.max().item())
    return _get_ext().q16_count_distinct_launcher(group_ids, supp_ids, num_groups, max_supp)

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
    Description: TPC-H Q16 — filter part table, identify and exclude bad suppliers, 
                 then calculate distinct supplier count per (brand, type, size) group.
                 Uses a fused CUDA kernel for atomic bitmask-based distinct counts.
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
    sizes_series = cudf.Series(list(sizes), dtype="int32")
    part_mask = (
        (df_part["p_brand"] != excluded_brand)
        & (~df_part["p_type"].str.startswith(excluded_type_prefix))
        & (df_part["p_size"].isin(sizes_series))
    )
    part_filtered = df_part.loc[part_mask, ["p_partkey", "p_brand", "p_type", "p_size"]]

    # --- Bad suppliers (subquery) ---
    bad_suppliers = df_supplier.loc[
        df_supplier["s_comment"].str.match(
            bad_supplier_comment_pattern.replace("%", ".*")
        ),
        "s_suppkey"
    ]

    # --- Filter partsupp: exclude bad suppliers ---
    ps_filtered = df_partsupp.loc[
        ~df_partsupp["ps_suppkey"].isin(bad_suppliers),
        ["ps_partkey", "ps_suppkey"]
    ]

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
