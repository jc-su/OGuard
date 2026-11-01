import os
import tempfile
from typing import Any

import cudf
import cupy as cp
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void q2_mark_i64_keys_kernel(
    const int64_t* __restrict__ keys,
    uint8_t* __restrict__ bitmap,
    int64_t n_keys,
    int64_t bitmap_size
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_keys) {
        int64_t k = keys[idx];
        if (k >= 0 && k < bitmap_size) {
            bitmap[k] = static_cast<uint8_t>(1);
        }
    }
}

__global__ void q2_membership_mask_i64_kernel(
    const int64_t* __restrict__ keys,
    const uint8_t* __restrict__ bitmap,
    bool* __restrict__ out,
    int64_t n,
    int64_t bitmap_size
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t k = keys[idx];
        out[idx] = (k >= 0 && k < bitmap_size && bitmap[k] != static_cast<uint8_t>(0));
    }
}

__global__ void q2_eq_i64_mask_kernel(
    const int64_t* __restrict__ values,
    int64_t target,
    bool* __restrict__ out,
    int64_t n
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (values[idx] == target);
    }
}

static torch::Tensor ensure_cuda_i64(torch::Tensor t, const char* name) {
    if (!t.is_cuda()) {
        t = t.to(torch::kCUDA);
    }
    TORCH_CHECK(t.scalar_type() == torch::kInt64, name, " must be int64");
    return t.contiguous();
}

torch::Tensor q2_build_bitmap_launcher(torch::Tensor keys, int64_t bitmap_size) {
    keys = ensure_cuda_i64(keys, "keys");
    TORCH_CHECK(bitmap_size >= 0, "bitmap_size must be non-negative");
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(keys.device());
    auto bitmap = torch::zeros({bitmap_size}, opts);

    int64_t n = keys.numel();
    if (n > 0 && bitmap_size > 0) {
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q2_mark_i64_keys_kernel<<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            keys.data_ptr<int64_t>(),
            bitmap.data_ptr<uint8_t>(),
            n,
            bitmap_size
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return bitmap;
}

torch::Tensor q2_membership_mask_launcher(torch::Tensor keys, torch::Tensor bitmap) {
    keys = ensure_cuda_i64(keys, "keys");
    if (!bitmap.is_cuda()) {
        bitmap = bitmap.to(torch::kCUDA);
    }
    TORCH_CHECK(bitmap.scalar_type() == torch::kUInt8, "bitmap must be uint8");
    bitmap = bitmap.contiguous();

    int64_t n = keys.numel();
    int64_t bitmap_size = bitmap.numel();
    auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kBool).device(keys.device()));

    if (n > 0) {
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q2_membership_mask_i64_kernel<<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            keys.data_ptr<int64_t>(),
            bitmap.data_ptr<uint8_t>(),
            out.data_ptr<bool>(),
            n,
            bitmap_size
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

torch::Tensor q2_eq_i64_mask_launcher(torch::Tensor values, int64_t target) {
    values = ensure_cuda_i64(values, "values");

    int64_t n = values.numel();
    auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kBool).device(values.device()));

    if (n > 0) {
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q2_eq_i64_mask_kernel<<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            values.data_ptr<int64_t>(),
            target,
            out.data_ptr<bool>(),
            n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q2_build_bitmap_launcher", &q2_build_bitmap_launcher, "TPC-H Q2 build int64 key bitmap (CUDA)");
    m.def("q2_membership_mask_launcher", &q2_membership_mask_launcher, "TPC-H Q2 int64 membership mask (CUDA)");
    m.def("q2_eq_i64_mask_launcher", &q2_eq_i64_mask_launcher, "TPC-H Q2 int64 equality mask (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension from the in-module .cu string.

    The CUDA code runs the bulk numeric scans used by Q2: bitmap construction,
    membership filtering of the 8M-row partsupp table, and p_size filtering.
    These are memory-bound passes, so the kernels use coalesced 1-D reads and
    write compact boolean masks consumed by cuDF, avoiding slower Python-side
    key-list materialization for the largest filter.
    """
    tmpdir = tempfile.mkdtemp(prefix="q02_cuda_")
    cu_path = os.path.join(tmpdir, "_q02_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q02_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _series_to_i64_tensor(s: cudf.Series) -> torch.Tensor:
    """Return a CUDA int64 torch tensor view/copy of a cuDF numeric column."""
    return torch.as_tensor(s.to_cupy(), dtype=torch.int64)


def _torch_bool_mask_to_cudf(mask: torch.Tensor, index: Any) -> cudf.Series:
    """Convert a CUDA bool tensor mask into a cuDF Series aligned to an index.

    The mask is produced by hand-written CUDA. A stream synchronize is used
    before handing the allocation to cuDF/CuPy so downstream libcudf filtering
    observes the completed kernel writes.
    """
    if mask.is_cuda:
        torch.cuda.synchronize(mask.device)
    arr = cp.from_dlpack(mask.contiguous())
    return cudf.Series(arr, index=index)


def _empty_result(
    df_supplier: cudf.DataFrame,
    df_nation: cudf.DataFrame,
    df_part: cudf.DataFrame,
) -> cudf.DataFrame:
    """Build an empty result with the exact SELECT column order and dtypes."""
    out = cudf.DataFrame()
    out["s_acctbal"] = df_supplier["s_acctbal"].head(0)
    out["s_name"] = df_supplier["s_name"].head(0)
    out["n_name"] = df_nation["n_name"].head(0)
    out["p_partkey"] = df_part["p_partkey"].head(0)
    out["p_mfgr"] = df_part["p_mfgr"].head(0)
    out["s_address"] = df_supplier["s_address"].head(0)
    out["s_phone"] = df_supplier["s_phone"].head(0)
    out["s_comment"] = df_supplier["s_comment"].head(0)
    return out.reset_index(drop=True)


def _cuda_membership_mask(
    keys: cudf.Series,
    member_keys: cudf.Series,
    bitmap_size: int,
) -> torch.Tensor:
    """Run the CUDA bitmap-membership filter for int64 keys.

    CUDA builds a dense byte bitmap for the relatively small supplier key
    domain, then scans the large keys column once with coalesced loads. This
    replaces a large cuDF isin/list filter in the hot path and minimizes
    temporary columns.
    """
    member_t = _series_to_i64_tensor(member_keys)
    bitmap_t = _EXT.q2_build_bitmap_launcher(member_t, int(bitmap_size))
    keys_t = _series_to_i64_tensor(keys)
    return _EXT.q2_membership_mask_launcher(keys_t, bitmap_t)


def _cuda_eq_i64_mask(values: cudf.Series, target: int) -> torch.Tensor:
    """Run a coalesced CUDA equality scan for an int64 cuDF column."""
    values_t = _series_to_i64_tensor(values)
    return _EXT.q2_eq_i64_mask_launcher(values_t, int(target))


def run_query(
    tables: dict[str, cudf.DataFrame],
    size_val: int = 15,
    type_suffix: str = "BRASS",
    region_name: str = "EUROPE",
    limit: int = 100,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr,
               s_address, s_phone, s_comment
        FROM part, supplier, partsupp, nation, region
        WHERE p_partkey = ps_partkey
          AND s_suppkey = ps_suppkey
          AND p_size = 15
          AND p_type LIKE '%BRASS'
          AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey
          AND r_name = 'EUROPE'
          AND ps_supplycost = (
              SELECT min(ps_supplycost)
              FROM partsupp, supplier, nation, region
              WHERE p_partkey = ps_partkey
                AND s_suppkey = ps_suppkey
                AND s_nationkey = n_nationkey
                AND n_regionkey = r_regionkey
                AND r_name = 'EUROPE')
        ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
        LIMIT 100;

    Description:
        TPC-H Q2. cuDF performs string predicates, joins, groupby-min, final
        projection, and ordering. Hand-written CUDA accelerates the memory-bound
        numeric scan steps: filtering the 8M-row partsupp table to European
        suppliers via a dense supplier-key bitmap, and filtering part.p_size.
        This avoids Python to_pylist/isin round-trips on the largest predicate,
        uses coalesced GPU reads, and reduces intermediate work before cuDF
        joins and aggregation.

    Params:
        tables: Mapping of TPC-H table names to cuDF DataFrames.
        size_val: Part size predicate value.
        type_suffix: Suffix used for p_type LIKE '%suffix'.
        region_name: Region name predicate, normally 'EUROPE'.
        limit: Maximum number of sorted rows to return.
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_partsupp = tables["partsupp"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    region_f = df_region.loc[
        (df_region["r_name"] == region_name).fillna(False),
        ["r_regionkey"],
    ]
    if len(region_f) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    nation_eu = df_nation.merge(
        region_f,
        left_on="n_regionkey",
        right_on="r_regionkey",
        how="inner",
    )[["n_nationkey", "n_name"]].reset_index(drop=True)
    if len(nation_eu) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    supp_eu_named = df_supplier.merge(
        nation_eu,
        left_on="s_nationkey",
        right_on="n_nationkey",
        how="inner",
    )[
        [
            "s_suppkey",
            "s_name",
            "s_address",
            "s_phone",
            "s_acctbal",
            "s_comment",
            "s_nationkey",
            "n_nationkey",
            "n_name",
        ]
    ].reset_index(drop=True)
    if len(supp_eu_named) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    max_supp = int(max(df_supplier["s_suppkey"].max(), df_partsupp["ps_suppkey"].max()))
    ps_mask_t = _cuda_membership_mask(
        df_partsupp["ps_suppkey"],
        supp_eu_named["s_suppkey"],
        max_supp + 1,
    )
    ps_mask = _torch_bool_mask_to_cudf(ps_mask_t, df_partsupp.index)

    ps_eu = df_partsupp.loc[
        ps_mask,
        ["ps_partkey", "ps_suppkey", "ps_supplycost"],
    ].reset_index(drop=True)
    if len(ps_eu) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    min_cost = ps_eu.groupby("ps_partkey", as_index=False)["ps_supplycost"].min()
    min_cost = min_cost.rename(columns={"ps_supplycost": "min_supplycost"}).reset_index(drop=True)

    size_mask_t = _cuda_eq_i64_mask(df_part["p_size"], int(size_val))
    size_mask = _torch_bool_mask_to_cudf(size_mask_t, df_part.index)

    parts_size = df_part.loc[
        size_mask,
        ["p_partkey", "p_mfgr", "p_type"],
    ]
    parts_f = parts_size.loc[
        parts_size["p_type"].str.endswith(type_suffix).fillna(False),
        ["p_partkey", "p_mfgr"],
    ].reset_index(drop=True)
    if len(parts_f) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    parts_min = parts_f.merge(
        min_cost,
        left_on="p_partkey",
        right_on="ps_partkey",
        how="inner",
    ).reset_index(drop=True)
    if len(parts_min) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    pm = parts_min[["ps_partkey", "p_partkey", "p_mfgr", "min_supplycost"]].reset_index(drop=True)

    cand = ps_eu.merge(pm, on="ps_partkey", how="inner").reset_index(drop=True)
    cand = cand.loc[cand["ps_supplycost"] == cand["min_supplycost"]].reset_index(drop=True)
    if len(cand) == 0:
        return _empty_result(df_supplier, df_nation, df_part)

    cand_clean = cand[["ps_suppkey", "p_partkey", "p_mfgr"]].reset_index(drop=True)

    result = cand_clean.merge(
        supp_eu_named,
        left_on="ps_suppkey",
        right_on="s_suppkey",
        how="inner",
    ).reset_index(drop=True)

    final = cudf.DataFrame()
    final["s_acctbal"] = result["s_acctbal"].reset_index(drop=True)
    final["s_name"] = result["s_name"].reset_index(drop=True)
    final["n_name"] = result["n_name"].reset_index(drop=True)
    final["p_partkey"] = result["p_partkey"].reset_index(drop=True)
    final["p_mfgr"] = result["p_mfgr"].reset_index(drop=True)
    final["s_address"] = result["s_address"].reset_index(drop=True)
    final["s_phone"] = result["s_phone"].reset_index(drop=True)
    final["s_comment"] = result["s_comment"].reset_index(drop=True)

    final = final.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    k = max(0, min(int(limit), len(final)))
    return final.iloc[:k].reset_index(drop=True)
