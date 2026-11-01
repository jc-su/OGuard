import os
import tempfile
from typing import Any

import cudf
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define EMPTY_KEY 0xffffffffffffffffULL

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__global__ void fill_i64_kernel(
    int64_t* __restrict__ ptr,
    int64_t n,
    int64_t value
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = value;
    }
}

__global__ void q16_insert_count_distinct_kernel(
    const int64_t* __restrict__ group_ids,
    const int64_t* __restrict__ supp_ids,
    int64_t n,
    unsigned long long* __restrict__ hash_table,
    uint64_t hash_mask,
    int64_t* __restrict__ counts,
    int64_t num_groups
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int64_t gid_i64 = group_ids[idx];
    int64_t sid_i64 = supp_ids[idx];

    if (gid_i64 < 0 || gid_i64 >= num_groups) {
        return;
    }

    uint64_t gid = static_cast<uint64_t>(gid_i64);
    uint64_t sid = static_cast<uint64_t>(static_cast<uint32_t>(sid_i64));
    uint64_t key = (gid << 32) | sid;

    if (key == EMPTY_KEY) {
        return;
    }

    uint64_t pos = mix64(key) & hash_mask;

    while (true) {
        unsigned long long old = atomicCAS(
            &hash_table[pos],
            static_cast<unsigned long long>(EMPTY_KEY),
            static_cast<unsigned long long>(key)
        );

        if (old == EMPTY_KEY) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(counts + gid_i64),
                static_cast<unsigned long long>(1)
            );
            return;
        }

        if (old == key) {
            return;
        }

        pos = (pos + 1ULL) & hash_mask;
    }
}

torch::Tensor q16_count_distinct_launcher(
    torch::Tensor group_ids,
    torch::Tensor supp_ids,
    int64_t num_groups
) {
    TORCH_CHECK(group_ids.scalar_type() == torch::kInt64, "group_ids must be int64");
    TORCH_CHECK(supp_ids.scalar_type() == torch::kInt64, "supp_ids must be int64");
    TORCH_CHECK(group_ids.is_cuda(), "group_ids must be a CUDA tensor");
    TORCH_CHECK(supp_ids.is_cuda(), "supp_ids must be a CUDA tensor");
    TORCH_CHECK(group_ids.numel() == supp_ids.numel(), "group_ids and supp_ids length mismatch");
    TORCH_CHECK(num_groups >= 0, "num_groups must be non-negative");

    group_ids = group_ids.contiguous();
    supp_ids = supp_ids.contiguous();

    auto options = group_ids.options().dtype(torch::kInt64);
    torch::Tensor counts = torch::empty({num_groups}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (num_groups > 0) {
        int count_blocks = static_cast<int>((num_groups + BLOCK_SIZE - 1) / BLOCK_SIZE);
        fill_i64_kernel<<<count_blocks, BLOCK_SIZE, 0, stream>>>(
            counts.data_ptr<int64_t>(),
            num_groups,
            static_cast<int64_t>(0)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int64_t n = group_ids.numel();
    if (n == 0 || num_groups == 0) {
        return counts;
    }

    uint64_t target = static_cast<uint64_t>(n) * 2ULL;
    if (target < 16ULL) {
        target = 16ULL;
    }

    uint64_t hash_size = 1ULL;
    while (hash_size < target) {
        hash_size <<= 1ULL;
    }

    TORCH_CHECK(hash_size <= static_cast<uint64_t>(INT64_MAX), "hash table too large");

    torch::Tensor hash_table = torch::empty({static_cast<int64_t>(hash_size)}, options);

    int table_blocks = static_cast<int>((static_cast<int64_t>(hash_size) + BLOCK_SIZE - 1) / BLOCK_SIZE);
    fill_i64_kernel<<<table_blocks, BLOCK_SIZE, 0, stream>>>(
        hash_table.data_ptr<int64_t>(),
        static_cast<int64_t>(hash_size),
        static_cast<int64_t>(-1)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int insert_blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q16_insert_count_distinct_kernel<<<insert_blocks, BLOCK_SIZE, 0, stream>>>(
        group_ids.data_ptr<int64_t>(),
        supp_ids.data_ptr<int64_t>(),
        n,
        reinterpret_cast<unsigned long long*>(hash_table.data_ptr<int64_t>()),
        hash_size - 1ULL,
        counts.data_ptr<int64_t>(),
        num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return counts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "q16_count_distinct_launcher",
        &q16_count_distinct_launcher,
        "TPC-H Q16 count distinct suppliers per group using CUDA open-addressed hash table"
    );
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension once at module import from the embedded .cu source."""
    tmpdir = tempfile.mkdtemp(prefix="q16_cuda_")
    cu_path = os.path.join(tmpdir, "_q16_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q16_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _query_core(
    group_ids: torch.Tensor,
    supp_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Run on GPU via the hand-written CUDA extension.

    Counts distinct ps_suppkey per integer group id by inserting composite
    (group_id, supplier_id) keys into a CUDA open-addressed hash table and
    incrementing the group count only for the winning insert. This replaces the
    reference torch.stack -> torch.unique(dim=0) -> scatter path with one custom
    GPU hash pass plus initialization, avoiding large sorted/unique temporaries.
    """
    return _EXT.q16_count_distinct_launcher(group_ids, supp_ids, int(num_groups))


def _empty_result() -> cudf.DataFrame:
    """Build the exact empty cuDF result shape used by the reference plan."""
    return cudf.DataFrame(
        {
            "p_brand": cudf.Series([], dtype="object"),
            "p_type": cudf.Series([], dtype="object"),
            "p_size": cudf.Series([], dtype="int32"),
            "supplier_cnt": cudf.Series([], dtype="int64"),
        }
    )


def _to_torch_int64(series: cudf.Series) -> torch.Tensor:
    """Expose a cuDF numeric column to PyTorch as an int64 CUDA tensor without a CPU copy."""
    return torch.as_tensor(series.to_cupy(), dtype=torch.int64)


def run_query(
    tables: dict[str, cudf.DataFrame],
    excluded_brand: str = "Brand#45",
    excluded_type_prefix: str = "MEDIUM POLISHED",
    sizes: tuple[int, ...] = (49, 14, 23, 45, 19, 3, 36, 9),
    bad_supplier_comment_pattern: str = "%Customer%Complaints%",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT
            p_brand,
            p_type,
            p_size,
            count(DISTINCT ps_suppkey) AS supplier_cnt
        FROM
            partsupp,
            part
        WHERE
            p_partkey = ps_partkey
            AND p_brand <> 'Brand#45'
            AND p_type NOT LIKE 'MEDIUM POLISHED%'
            AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
            AND ps_suppkey NOT IN (
                SELECT s_suppkey
                FROM supplier
                WHERE s_comment LIKE '%Customer%Complaints%')
        GROUP BY
            p_brand,
            p_type,
            p_size
        ORDER BY
            supplier_cnt DESC,
            p_brand,
            p_type,
            p_size;

    Description:
        TPC-H Q16. cuDF performs string predicates, the supplier-comment
        subquery, the part/partsupp join, and final sorting because those stages
        depend on string columns and DataFrame semantics. The hot numeric
        distinct-count step runs in hand-written CUDA: joined rows are converted
        to integer group ids and supplier ids, then a fused CUDA hash-table
        kernel deduplicates (group, supplier) pairs and counts each group. This
        should be faster than the reference PyTorch path because it avoids
        torch.stack, torch.unique(dim=0), scatter temporaries, and general
        sort/unique machinery for this fixed two-int64 distinct aggregation.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables containing
            "part", "partsupp", and "supplier".
        excluded_brand (str): Brand to exclude from part rows.
        excluded_type_prefix (str): Prefix excluded by p_type NOT LIKE 'prefix%'.
        sizes (tuple[int, ...]): Allowed p_size values.
        bad_supplier_comment_pattern (str): SQL LIKE-style pattern used by the
            reference as regex after replacing '%' with '.*'.
    """
    df_part = tables["part"]
    df_partsupp = tables["partsupp"]
    df_supplier = tables["supplier"]

    part_mask = (
        (df_part["p_brand"] != excluded_brand)
        & (~df_part["p_type"].str.startswith(excluded_type_prefix))
        & (df_part["p_size"].isin(list(sizes)))
    )
    part_filtered = df_part.loc[
        part_mask, ["p_partkey", "p_brand", "p_type", "p_size"]
    ]

    if len(part_filtered) == 0:
        return _empty_result()

    bad_regex = bad_supplier_comment_pattern.replace("%", ".*")
    bad_suppliers = df_supplier.loc[
        df_supplier["s_comment"].str.match(bad_regex), "s_suppkey"
    ]

    joined = part_filtered.merge(
        df_partsupp[["ps_partkey", "ps_suppkey"]],
        left_on="p_partkey",
        right_on="ps_partkey",
    )

    if len(joined) == 0:
        return _empty_result()

    joined = joined.loc[
        ~joined["ps_suppkey"].isin(bad_suppliers),
        ["p_brand", "p_type", "p_size", "ps_suppkey"],
    ]

    if len(joined) == 0:
        return _empty_result()

    group_cols = ["p_brand", "p_type", "p_size"]
    group_df = joined[group_cols].drop_duplicates().reset_index(drop=True)
    group_df["_gid"] = cudf.Series(range(len(group_df)), dtype="int64")

    joined_with_gid = joined.merge(group_df, on=group_cols)

    num_groups = len(group_df)
    if num_groups == 0:
        return _empty_result()

    group_ids_t = _to_torch_int64(joined_with_gid["_gid"])
    supp_ids_t = _to_torch_int64(joined_with_gid["ps_suppkey"])

    counts_t = _query_core(group_ids_t, supp_ids_t, num_groups)
    torch.cuda.current_stream(counts_t.device).synchronize()

    result = cudf.DataFrame(
        {
            "p_brand": group_df["p_brand"],
            "p_type": group_df["p_type"],
            "p_size": group_df["p_size"],
            "supplier_cnt": cudf.Series(counts_t, dtype="int64"),
        }
    )

    result = result.sort_values(
        by=["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    return result
