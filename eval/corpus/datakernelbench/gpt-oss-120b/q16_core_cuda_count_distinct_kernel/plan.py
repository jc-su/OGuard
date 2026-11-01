import os
import tempfile
import torch
import cudf
import pandas as pd
from typing import Tuple
from torch.utils.cpp_extension import load

# --------------------------------------------------------------
# CUDA source: fused distinct-count kernel + launcher
# --------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#define BLOCK_SIZE 1024

// ----------------------------------------------------------
// Kernel: after sorting (group, supp) pairs, each thread
// checks if its element is the start of a new distinct pair
// and atomically increments the per‑group counter.
// ----------------------------------------------------------
__global__ void count_distinct_kernel(
    const int64_t* __restrict__ group,
    const int64_t* __restrict__ supp,
    int64_t n,
    unsigned long long* __restrict__ out)          // use 64‑bit unsigned for atomicAdd
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bool is_unique = false;
    if (idx == 0) {
        is_unique = true;
    } else {
        int64_t g0 = group[idx];
        int64_t s0 = supp[idx];
        int64_t g1 = group[idx - 1];
        int64_t s1 = supp[idx - 1];
        if (g0 != g1 || s0 != s1) {
            is_unique = true;
        }
    }

    if (is_unique) {
        // one atomic per distinct pair
        atomicAdd(&out[group[idx]], 1ULL);
    }
}

// ----------------------------------------------------------
// Launcher: sorts the (group, supp) pairs, then runs the
// distinct‑count kernel. Returns a torch::Tensor of int64
// counts per group (the atomic writes happen on the same
// memory, just viewed as unsigned long long inside the kernel).
// ----------------------------------------------------------
torch::Tensor distinct_count_launcher(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    int64_t num_groups)
{
    TORCH_CHECK(group_ids.scalar_type() == torch::kInt64,
                "group_ids must be int64");
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64,
                "supp_ids must be int64");
    TORCH_CHECK(group_ids.is_cuda() && supp_ids.is_cuda(),
                "inputs must be on CUDA");

    // Ensure contiguous memory layout
    group_ids = group_ids.contiguous();
    supp_ids  = supp_ids.contiguous();

    int64_t n = group_ids.numel();

    // Temporary copies that will be sorted in‑place
    torch::Tensor sorted_group = torch::empty_like(group_ids);
    torch::Tensor sorted_supp  = torch::empty_like(supp_ids);
    sorted_group.copy_(group_ids);
    sorted_supp.copy_(supp_ids);

    // ------------------------------------------------------
    // Thrust sort by (group, supp) using zip iterator
    // ------------------------------------------------------
    thrust::device_ptr<int64_t> d_grp   = thrust::device_pointer_cast(sorted_group.data_ptr<int64_t>());
    thrust::device_ptr<int64_t> d_supp  = thrust::device_pointer_cast(sorted_supp.data_ptr<int64_t>());
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_grp, d_supp));
    thrust::sort(zip_begin, zip_begin + n);

    // Allocate output vector (int64) and zero‑initialize
    torch::Tensor out = torch::zeros({num_groups},
        torch::TensorOptions().dtype(torch::kInt64).device(group_ids.device()));

    // Launch kernel
    int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_distinct_kernel<<<static_cast<unsigned int>(blocks), BLOCK_SIZE>>>(
        sorted_group.data_ptr<int64_t>(),
        sorted_supp.data_ptr<int64_t>(),
        n,
        reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false,
            "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}

// ----------------------------------------------------------
// PyBind entry point
// ----------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("distinct_count_launcher",
          &distinct_count_launcher,
          "Count distinct ps_suppkey per (brand,type,size) group (CUDA)");
}
'''

def _load_cuda():
    """
    Compile the embedded CUDA source and load it as a PyTorch extension.
    The temporary directory is unique per import to avoid name clashes.
    """
    tmpdir = tempfile.mkdtemp(prefix="tpch_q16_cuda_")
    cu_path = os.path.join(tmpdir, "q16_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="tpch_q16_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

# Load the compiled extension once at module import time
_EXT = _load_cuda()

def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    CUDA‑fused distinct‑count per group.

    Replaces the reference eager implementation that used
    ``torch.unique`` + ``scatter_add`` with a single kernel that:

    1. Sorts ``(group, supp)`` pairs (coalesced 64‑bit loads).
    2. Scans the sorted array once, detecting the start of each distinct pair.
    3. Atomically increments the per‑group counter (one atomic per distinct pair,
       not per row).

    This reduces memory traffic (single pass over sorted data) and
    eliminates the Python‑side temporary tensors created by ``torch.unique``.
    """
    if not group_ids.is_cuda:
        group_ids = group_ids.cuda()
    if not supp_ids.is_cuda:
        supp_ids = supp_ids.cuda()
    group_ids = group_ids.contiguous()
    supp_ids = supp_ids.contiguous()
    return _EXT.distinct_count_launcher(group_ids, supp_ids, num_groups)

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
    Description: TPC‑H Q16 — count distinct suppliers per (brand, type, size)
                 with filters on part attributes and excluding suppliers whose
                 comment matches a complaint pattern.
    Params:
        excluded_brand (str): brand to exclude.
        excluded_type_prefix (str): type prefix to exclude (LIKE 'prefix%').
        sizes (tuple[int, ...]): allowed p_size values.
        bad_supplier_comment_pattern (str): SQL LIKE pattern for supplier
                                            comments to exclude.
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
