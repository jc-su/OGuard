import os
import tempfile
import torch
import cudf
import pandas as pd
from typing import Tuple
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

static constexpr unsigned long long EMPTY_KEY = 0xffffffffffffffffULL;

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

__global__ void q16_distinct_count_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t n_elements,
    int64_t num_groups,
    unsigned long long* __restrict__ table,
    int64_t table_capacity,
    int64_t* __restrict__ counts
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_elements) {
        return;
    }

    int64_t gid64 = group_ids[idx];
    int64_t supp64 = supp_ids[idx];

    if (gid64 < 0 || gid64 >= num_groups) {
        return;
    }

    unsigned long long gid = static_cast<unsigned long long>(static_cast<uint32_t>(gid64));
    unsigned long long supp = static_cast<unsigned long long>(static_cast<uint32_t>(supp64));
    unsigned long long key = (gid << 32) | supp;

    if (key == EMPTY_KEY) {
        key = 0xfffffffffffffffeULL;
    }

    unsigned long long mask = static_cast<unsigned long long>(table_capacity - 1);
    unsigned long long slot = mix64(key) & mask;

    for (int64_t probe = 0; probe < table_capacity; ++probe) {
        unsigned long long old = atomicCAS(&table[slot], EMPTY_KEY, key);
        if (old == EMPTY_KEY) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&counts[gid64]), 1ULL);
            return;
        }
        if (old == key) {
            return;
        }
        slot = (slot + 1ULL) & mask;
    }
}

torch::Tensor q16_distinct_count_launcher(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    int64_t num_groups
) {
    TORCH_CHECK(group_ids.scalar_type() == torch::kInt64, "group_ids must be int64");
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64, "supp_ids must be int64");
    TORCH_CHECK(group_ids.numel() == supp_ids.numel(), "group_ids and supp_ids must have same length");
    TORCH_CHECK(num_groups >= 0, "num_groups must be non-negative");

    if (!group_ids.is_cuda()) {
        group_ids = group_ids.cuda();
    }
    if (!supp_ids.is_cuda()) {
        supp_ids = supp_ids.cuda();
    }

    group_ids = group_ids.contiguous();
    supp_ids = supp_ids.contiguous();

    int64_t n = group_ids.numel();
    auto options = group_ids.options().dtype(torch::kInt64);
    torch::Tensor counts = torch::empty({num_groups}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (num_groups > 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(
            counts.data_ptr<int64_t>(),
            0,
            static_cast<size_t>(num_groups) * sizeof(int64_t),
            stream
        ));
    }

    if (n == 0 || num_groups == 0) {
        return counts;
    }

    int64_t min_capacity = 16;
    int64_t needed = n * 4;
    if (needed > min_capacity) {
        min_capacity = needed;
    }

    int64_t table_capacity = 1;
    while (table_capacity < min_capacity) {
        table_capacity <<= 1;
    }

    torch::Tensor table_tensor = torch::empty({table_capacity}, options);
    C10_CUDA_CHECK(cudaMemsetAsync(
        table_tensor.data_ptr<int64_t>(),
        0xFF,
        static_cast<size_t>(table_capacity) * sizeof(int64_t),
        stream
    ));

    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q16_distinct_count_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        group_ids.data_ptr<int64_t>(),
        supp_ids.data_ptr<int64_t>(),
        n,
        num_groups,
        reinterpret_cast<unsigned long long*>(table_tensor.data_ptr<int64_t>()),
        table_capacity,
        counts.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "q16_distinct_count_launcher",
        &q16_distinct_count_launcher,
        "TPC-H Q16 distinct supplier count per integer group (CUDA)"
    );
}
'''


def _load_cuda():
    """
    Compile the CUDA extension from the embedded .cu source at import time.

    The device code performs the numeric hot path for Q16: a GPU open-addressing
    hash table over packed (group_id, ps_suppkey) pairs and an atomic increment
    only when a distinct pair is first inserted. This replaces torch.stack +
    torch.unique + scatter_add with one CUDA pass plus table initialization,
    avoiding large temporary pair tensors and the general-purpose unique sort.
    """
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


def _launch_q16_distinct_count(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Launch the CUDA hot path for distinct pair counting.

    cuDF remains responsible for string filtering, joins, and group-id creation in
    run_query. This helper runs on GPU tensors only and uses a custom CUDA hash
    table to count each unique (group_id, supplier_id) pair once, which should be
    faster than the reference PyTorch unique-based path for this memory-bound
    distinct-count workload.
    """
    return _EXT.q16_distinct_count_launcher(group_ids, supp_ids, int(num_groups))


def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Count distinct ps_suppkey per group on the GPU with a hand-written CUDA kernel.

    group_ids: int64 tensor in [0, num_groups)
    supp_ids: int64 tensor of ps_suppkey values (already filtered).

    The CUDA extension packs each (group_id, supp_id) pair into a 64-bit key,
    inserts keys into a GPU open-addressing hash table, and atomically increments
    the group's int64 count only for the first insertion. This avoids the
    reference torch.stack + torch.unique(dim=0) + scatter_add sequence and its
    large temporaries/general sort, reducing the hot-path work to one fused
    distinct-count pass over compact integer tensors.
    """
    return _launch_q16_distinct_count(group_ids, supp_ids, num_groups)


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
