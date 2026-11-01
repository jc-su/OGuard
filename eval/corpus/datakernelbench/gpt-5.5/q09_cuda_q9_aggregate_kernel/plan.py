import os
import tempfile
from types import ModuleType
from typing import Any

import cudf
import numpy as np
import pandas as pd
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#define BLOCK_SIZE 256
#define MAX_GROUPS 2048
#define MAX_GRID_BLOCKS 4096

__global__ void q9_aggregate_kernel(
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    const int64_t* __restrict__ nationkey,
    const int32_t* __restrict__ year,
    int64_t n_elements,
    int32_t year_min,
    int32_t n_years,
    int64_t max_nation_key,
    double* __restrict__ out_sums,
    unsigned long long* __restrict__ out_counts
) {
    __shared__ double sh_sums[MAX_GROUPS];
    __shared__ unsigned long long sh_counts[MAX_GROUPS];

    const int tid = threadIdx.x;
    const int group_count = static_cast<int>((max_nation_key + 1) * static_cast<int64_t>(n_years));

    for (int g = tid; g < group_count; g += blockDim.x) {
        sh_sums[g] = 0.0;
        sh_counts[g] = 0ULL;
    }
    __syncthreads();

    const int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_elements;
         idx += stride) {
        const int64_t nk = nationkey[idx];
        const int32_t yr = year[idx];
        const int32_t yoff = yr - year_min;

        if (nk >= 0 && nk <= max_nation_key && yoff >= 0 && yoff < n_years) {
            const int64_t group64 = nk * static_cast<int64_t>(n_years) + static_cast<int64_t>(yoff);
            if (group64 >= 0 && group64 < group_count) {
                const int group = static_cast<int>(group64);
                const double amount =
                    extendedprice[idx] * (1.0 - discount[idx]) - supplycost[idx] * quantity[idx];
                atomicAdd(&sh_sums[group], amount);
                atomicAdd(&sh_counts[group], 1ULL);
            }
        }
    }

    __syncthreads();

    for (int g = tid; g < group_count; g += blockDim.x) {
        const unsigned long long c = sh_counts[g];
        if (c != 0ULL) {
            atomicAdd(&out_sums[g], sh_sums[g]);
            atomicAdd(&out_counts[g], c);
        }
    }
}

std::vector<torch::Tensor> q9_aggregate_launcher(
    torch::Tensor extendedprice,
    torch::Tensor discount,
    torch::Tensor supplycost,
    torch::Tensor quantity,
    torch::Tensor nationkey,
    torch::Tensor year,
    int32_t year_min,
    int32_t year_max,
    int64_t max_nation_key
) {
    TORCH_CHECK(extendedprice.scalar_type() == torch::kFloat64, "extendedprice must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64, "supplycost must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(nationkey.scalar_type() == torch::kInt64, "nationkey must be int64");
    TORCH_CHECK(year.scalar_type() == torch::kInt32, "year must be int32");
    TORCH_CHECK(year_max >= year_min, "year_max must be >= year_min");
    TORCH_CHECK(max_nation_key >= 0, "max_nation_key must be non-negative");

    if (!extendedprice.is_cuda()) extendedprice = extendedprice.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!supplycost.is_cuda()) supplycost = supplycost.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();
    if (!nationkey.is_cuda()) nationkey = nationkey.cuda();
    if (!year.is_cuda()) year = year.cuda();

    extendedprice = extendedprice.contiguous();
    discount = discount.contiguous();
    supplycost = supplycost.contiguous();
    quantity = quantity.contiguous();
    nationkey = nationkey.contiguous();
    year = year.contiguous();

    const int64_t n = extendedprice.numel();
    TORCH_CHECK(discount.numel() == n, "discount length mismatch");
    TORCH_CHECK(supplycost.numel() == n, "supplycost length mismatch");
    TORCH_CHECK(quantity.numel() == n, "quantity length mismatch");
    TORCH_CHECK(nationkey.numel() == n, "nationkey length mismatch");
    TORCH_CHECK(year.numel() == n, "year length mismatch");

    const int32_t n_years = year_max - year_min + 1;
    const int64_t group_count64 = (max_nation_key + 1) * static_cast<int64_t>(n_years);
    TORCH_CHECK(group_count64 > 0 && group_count64 <= MAX_GROUPS,
                "too many nation/year groups for q9 CUDA kernel");

    auto sum_options = extendedprice.options().dtype(torch::kFloat64);
    auto count_options = extendedprice.options().dtype(torch::kInt64);

    torch::Tensor out_sums = torch::zeros({group_count64}, sum_options);
    torch::Tensor out_counts = torch::zeros({group_count64}, count_options);

    if (n > 0) {
        const int64_t tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int blocks = static_cast<int>(std::min<int64_t>(tiles, MAX_GRID_BLOCKS));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        q9_aggregate_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            extendedprice.data_ptr<double>(),
            discount.data_ptr<double>(),
            supplycost.data_ptr<double>(),
            quantity.data_ptr<double>(),
            nationkey.data_ptr<int64_t>(),
            year.data_ptr<int32_t>(),
            n,
            year_min,
            n_years,
            max_nation_key,
            out_sums.data_ptr<double>(),
            reinterpret_cast<unsigned long long*>(out_counts.data_ptr<int64_t>())
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {out_sums, out_counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q9_aggregate_launcher", &q9_aggregate_launcher,
          "TPC-H Q9 fixed-small-group profit aggregation (CUDA)");
}
'''


_MAX_GROUPS: int = 2048


def _load_cuda() -> ModuleType:
    """
    Compile the CUDA extension from the in-module _CUDA_SOURCE string.

    Runs at Python import time on the host. The generated CUDA kernel performs the
    GPU hot path: amount projection plus fixed nation/year aggregation in one pass,
    avoiding a materialized amount column and a large cuDF groupby.
    """
    tmpdir = tempfile.mkdtemp(prefix="q9_cuda_")
    cu_path = os.path.join(tmpdir, "_q9_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)

    return load(
        name="q9_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT: ModuleType = _load_cuda()


def _series_to_torch_f64(series: cudf.Series) -> torch.Tensor:
    """
    Convert a cuDF numeric Series to a CUDA torch.float64 tensor.

    The data stays on GPU via the CUDA array interface. This feeds the hand-written
    CUDA aggregation kernel with the same float64 precision as the reference plan.
    """
    return torch.as_tensor(series.to_cupy(), dtype=torch.float64)


def _series_to_torch_i64(series: cudf.Series) -> torch.Tensor:
    """
    Convert a cuDF integer Series to a CUDA torch.int64 tensor.

    The conversion is used for nation keys consumed by the CUDA fixed-group
    aggregation kernel.
    """
    return torch.as_tensor(series.to_cupy(), dtype=torch.int64)


def _series_to_torch_i32(series: cudf.Series) -> torch.Tensor:
    """
    Convert a cuDF integer Series to a CUDA torch.int32 tensor.

    The conversion is used for extracted order years consumed by the CUDA
    fixed-group aggregation kernel.
    """
    return torch.as_tensor(series.to_cupy(), dtype=torch.int32)


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
    s_nationkey: torch.Tensor,
    o_year: torch.Tensor,
    year_min: int,
    year_max: int,
    max_nation_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the CUDA hot path for TPC-H Q9 after cuDF join preparation.

    The kernel fuses amount computation
    ``price * (1 - discount) - supplycost * quantity`` with aggregation into a
    small nation/year matrix. This is faster than the reference eager Torch
    projection plus cuDF groupby because it makes one coalesced GPU pass over the
    joined numeric columns, keeps the projection in registers, uses shared-memory
    block accumulation, and emits only one global atomic per non-empty block/group.
    """
    group_count = (max_nation_key + 1) * (year_max - year_min + 1)
    if group_count <= 0 or group_count > _MAX_GROUPS:
        raise ValueError("TPC-H Q9 CUDA kernel supports at most 2048 nation/year groups")

    out: list[torch.Tensor] = _EXT.q9_aggregate_launcher(
        l_extendedprice,
        l_discount,
        ps_supplycost,
        l_quantity,
        s_nationkey,
        o_year,
        int(year_min),
        int(year_max),
        int(max_nation_key),
    )
    return out[0], out[1]


def _empty_result(df_nation: cudf.DataFrame) -> cudf.DataFrame:
    """
    Build an empty cuDF result with the SQL output schema.

    This host/cuDF helper is used when the cuDF string filter or joins produce no
    rows, so no CUDA aggregation launch is necessary.
    """
    return cudf.DataFrame(
        {
            "nation": cudf.Series([], dtype=df_nation["n_name"].dtype),
            "o_year": cudf.Series([], dtype="int32"),
            "sum_profit": cudf.Series([], dtype="float64"),
        }
    )


def _build_result_from_agg(
    sums: torch.Tensor,
    counts: torch.Tensor,
    df_nation: cudf.DataFrame,
    year_min: int,
    year_max: int,
    max_nation_key: int,
) -> cudf.DataFrame:
    """
    Convert the small CUDA nation/year aggregate matrix into the final cuDF frame.

    The expensive aggregation has already run on GPU. This helper synchronizes only
    the tiny fixed-size aggregate arrays, maps nation keys to names, removes groups
    with zero rows, and applies the SQL ordering by nation ascending and year
    descending.
    """
    n_years = year_max - year_min + 1
    sums_np = np.asarray(sums.detach().cpu().numpy(), dtype=np.float64)
    counts_np = np.asarray(counts.detach().cpu().numpy(), dtype=np.int64)

    nation_pdf = df_nation[["n_nationkey", "n_name"]].to_pandas()
    key_to_name: dict[int, Any] = {
        int(k): v for k, v in zip(nation_pdf["n_nationkey"], nation_pdf["n_name"])
    }

    rows: list[tuple[Any, np.int32, np.float64]] = []
    for nk in range(max_nation_key + 1):
        name = key_to_name.get(nk)
        if name is None:
            continue
        base = nk * n_years
        for yoff in range(n_years):
            pos = base + yoff
            if counts_np[pos] > 0:
                rows.append((name, np.int32(year_min + yoff), np.float64(sums_np[pos])))

    if not rows:
        return _empty_result(df_nation)

    pdf = pd.DataFrame(rows, columns=["nation", "o_year", "sum_profit"])
    pdf["o_year"] = pdf["o_year"].astype(np.int32)
    pdf["sum_profit"] = pdf["sum_profit"].astype(np.float64)
    pdf = pdf.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)
    return cudf.DataFrame(pdf)


def run_query(tables: dict[str, cudf.DataFrame], name_pattern: str = "green") -> cudf.DataFrame:
    """
    SQL:
        SELECT nation, o_year, sum(amount) AS sum_profit
        FROM (
            SELECT
                n_name AS nation,
                extract(year FROM o_orderdate) AS o_year,
                l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
            FROM part, supplier, lineitem, partsupp, orders, nation
            WHERE
                s_suppkey = l_suppkey
                AND ps_suppkey = l_suppkey
                AND ps_partkey = l_partkey
                AND p_partkey = l_partkey
                AND o_orderkey = l_orderkey
                AND s_nationkey = n_nationkey
                AND p_name LIKE '%green%'
        ) AS profit
        GROUP BY nation, o_year
        ORDER BY nation, o_year DESC;

    Description:
        TPC-H Q9 - Product Type Profit Measure. cuDF performs the string
        predicate and relational joins, but the join order is rewritten to first
        filter partsupp by matching part names and attach supplier nation on that
        smaller dimension side. The GPU hot path is a hand-written CUDA kernel
        that fuses amount projection and nation/year sum aggregation. This avoids
        materializing a large amount column and avoids a large cuDF groupby,
        replacing them with one coalesced pass over joined numeric columns plus
        shared-memory block reductions and sparse global atomics.

    Params:
        tables: Mapping of TPC-H table names to cuDF DataFrames.
        name_pattern: Substring to filter part names; '%' characters are stripped
            to match the reference LIKE '%pattern%' behavior.
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    pat = name_pattern.replace("%", "")

    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]
    if len(parts_filtered) == 0:
        return _empty_result(df_nation)

    ps_filtered = df_partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]].merge(
        parts_filtered,
        left_on="ps_partkey",
        right_on="p_partkey",
    )[["ps_partkey", "ps_suppkey", "ps_supplycost"]]
    if len(ps_filtered) == 0:
        return _empty_result(df_nation)

    ps_supplier = ps_filtered.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="ps_suppkey",
        right_on="s_suppkey",
    )[["ps_partkey", "ps_suppkey", "ps_supplycost", "s_nationkey"]]
    if len(ps_supplier) == 0:
        return _empty_result(df_nation)

    lineitem_proj = df_lineitem[
        [
            "l_partkey",
            "l_suppkey",
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_quantity",
        ]
    ]

    li_ps_s = lineitem_proj.merge(
        ps_supplier,
        left_on=["l_partkey", "l_suppkey"],
        right_on=["ps_partkey", "ps_suppkey"],
    )[
        [
            "l_orderkey",
            "l_extendedprice",
            "l_discount",
            "l_quantity",
            "ps_supplycost",
            "s_nationkey",
        ]
    ]
    if len(li_ps_s) == 0:
        return _empty_result(df_nation)

    joined = li_ps_s.merge(
        df_orders[["o_orderkey", "o_orderdate"]],
        left_on="l_orderkey",
        right_on="o_orderkey",
    )[
        [
            "l_extendedprice",
            "l_discount",
            "l_quantity",
            "ps_supplycost",
            "s_nationkey",
            "o_orderdate",
        ]
    ]
    if len(joined) == 0:
        return _empty_result(df_nation)

    o_year_series = joined["o_orderdate"].dt.year.astype("int32")
    year_min = int(o_year_series.min())
    year_max = int(o_year_series.max())
    max_nation_key = int(df_nation["n_nationkey"].max())

    sums, counts = _query_core(
        _series_to_torch_f64(joined["l_extendedprice"]),
        _series_to_torch_f64(joined["l_discount"]),
        _series_to_torch_f64(joined["ps_supplycost"]),
        _series_to_torch_f64(joined["l_quantity"]),
        _series_to_torch_i64(joined["s_nationkey"]),
        _series_to_torch_i32(o_year_series),
        year_min,
        year_max,
        max_nation_key,
    )

    return _build_result_from_agg(
        sums=sums,
        counts=counts,
        df_nation=df_nation,
        year_min=year_min,
        year_max=year_max,
        max_nation_key=max_nation_key,
    )
