import os
import tempfile
from typing import Any

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;

#define BLOCK_SIZE 256
#define MAX_NATIONS 1024

__global__ void q5_aggregate_kernel(
    const int64_t* __restrict__ nation_id,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ out,
    int64_t n_elements,
    int64_t num_nations
) {
    __shared__ double group_sums[MAX_NATIONS];

    for (int64_t g = threadIdx.x; g < num_nations; g += blockDim.x) {
        group_sums[g] = 0.0;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t i = idx; i < n_elements; i += stride) {
        int64_t gid = nation_id[i];
        if (gid >= 0 && gid < num_nations) {
            double rev = price[i] * (1.0 - discount[i]);
            atomicAdd(&group_sums[gid], rev);
        }
    }

    __syncthreads();

    for (int64_t g = threadIdx.x; g < num_nations; g += blockDim.x) {
        atomicAdd(&out[g], group_sums[g]);
    }
}

__global__ void q5_sort_kernel(
    const double* __restrict__ revenue,
    int64_t* __restrict__ sorted_ids,
    double* __restrict__ sorted_revenue,
    int64_t num_nations
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int64_t i = 0; i < num_nations; ++i) {
        sorted_ids[i] = i;
        sorted_revenue[i] = revenue[i];
    }

    for (int64_t i = 1; i < num_nations; ++i) {
        double key_rev = sorted_revenue[i];
        int64_t key_id = sorted_ids[i];
        int64_t j = i - 1;

        while (
            j >= 0 &&
            (
                sorted_revenue[j] < key_rev ||
                (sorted_revenue[j] == key_rev && sorted_ids[j] > key_id)
            )
        ) {
            sorted_revenue[j + 1] = sorted_revenue[j];
            sorted_ids[j + 1] = sorted_ids[j];
            --j;
        }

        sorted_revenue[j + 1] = key_rev;
        sorted_ids[j + 1] = key_id;
    }
}

py::tuple q5_launcher(
    torch::Tensor nation_id,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    int64_t num_nations
) {
    TORCH_CHECK(num_nations >= 0, "num_nations must be non-negative");
    TORCH_CHECK(num_nations <= MAX_NATIONS, "num_nations exceeds MAX_NATIONS");
    TORCH_CHECK(nation_id.scalar_type() == torch::kInt64, "nation_id must be int64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(nation_id.numel() == l_extendedprice.numel(), "nation_id and l_extendedprice size mismatch");
    TORCH_CHECK(l_discount.numel() == l_extendedprice.numel(), "l_discount and l_extendedprice size mismatch");

    if (!l_extendedprice.is_cuda()) {
        l_extendedprice = l_extendedprice.to(torch::kCUDA);
    }
    if (!l_discount.is_cuda()) {
        l_discount = l_discount.to(l_extendedprice.device());
    }
    if (!nation_id.is_cuda()) {
        nation_id = nation_id.to(l_extendedprice.device());
    }

    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    nation_id = nation_id.contiguous();

    c10::cuda::CUDAGuard device_guard(l_extendedprice.device());

    auto revenue_options = l_extendedprice.options().dtype(torch::kFloat64);
    auto id_options = torch::TensorOptions().dtype(torch::kInt64).device(l_extendedprice.device());

    auto revenue = torch::zeros({num_nations}, revenue_options);
    auto sorted_revenue = torch::empty({num_nations}, revenue_options);
    auto sorted_ids = torch::empty({num_nations}, id_options);

    if (num_nations == 0) {
        return py::make_tuple(sorted_ids, sorted_revenue);
    }

    int64_t n = l_extendedprice.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (n > 0) {
        int64_t needed_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blocks = static_cast<int>(std::min<int64_t>(needed_blocks, 4096));
        q5_aggregate_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            nation_id.data_ptr<int64_t>(),
            l_extendedprice.data_ptr<double>(),
            l_discount.data_ptr<double>(),
            revenue.data_ptr<double>(),
            n,
            num_nations
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    q5_sort_kernel<<<1, 1, 0, stream>>>(
        revenue.data_ptr<double>(),
        sorted_ids.data_ptr<int64_t>(),
        sorted_revenue.data_ptr<double>(),
        num_nations
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return py::make_tuple(sorted_ids, sorted_revenue);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_launcher", &q5_launcher, "TPC-H Q5 grouped revenue aggregation and small descending sort (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension from an in-module .cu string.

    The numeric hot path runs in hand-written CUDA: one fused coalesced pass computes
    revenue and accumulates per nation through shared-memory block aggregation, then a
    tiny CUDA kernel sorts the small nation result. This avoids PyTorch's eager
    multiply temporary and row-level scatter_add global atomics.
    """
    tmpdir = tempfile.mkdtemp(prefix="q5_cuda_")
    cu_path = os.path.join(tmpdir, "_q5_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q5_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the Q5 numeric core on the GPU using the CUDA extension.

    cuDF still performs joins/filtering in ``run_query``. This function receives the
    compact joined columns as tensors and performs revenue projection, grouped
    aggregation, and the small descending sort in CUDA. It should be faster than the
    reference eager PyTorch region because it fuses multiply/subtract with aggregation,
    avoids materializing the full revenue vector, and replaces per-row global scatter
    atomics with per-block shared-memory accumulation plus one global atomic per group
    per block.
    """
    result = _EXT.q5_launcher(
        nation_id,
        l_extendedprice,
        l_discount,
        int(num_nations),
    )
    return result[0], result[1]


def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "ASIA",
    orderdate_lo: str = "1994-01-01",
    orderdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
         FROM customer, orders, lineitem, supplier, nation, region
         WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
           AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
           AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
           AND r_name = 'ASIA'
           AND o_orderdate >= DATE '1994-01-01'
           AND o_orderdate <  DATE '1995-01-01'
         GROUP BY n_name ORDER BY revenue DESC;
    Description: TPC-H Q5 — Local supplier volume for a region over one year.
    Params:
      region_name (str): region to filter on.
      orderdate_lo (str): inclusive lower bound on order date.
      orderdate_hi (str): exclusive upper bound on order date.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter region
    region_f = df_region[df_region["r_name"] == region_name]
    # Nation within region
    nation_f = df_nation.merge(region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey")

    # Orders filtered by date
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (orders_f["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = orders_f[
        (orders_f["o_orderdate_days"] >= lo_days)
        & (orders_f["o_orderdate_days"] < hi_days)
    ]

    # Join customer with orders on custkey
    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f[["o_orderkey", "o_custkey"]], left_on="c_custkey", right_on="o_custkey"
    )

    # Join with lineitem on orderkey
    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey", right_on="l_orderkey",
    )

    # Join with supplier on suppkey AND c_nationkey = s_nationkey
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey", right_on="s_suppkey",
    )
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    # Join with filtered nation on s_nationkey = n_nationkey
    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey", right_on="n_nationkey",
    )

    # Build nation_id mapping from distinct nations in filter
    nations_unique = nation_f[["n_nationkey", "n_name"]].drop_duplicates().sort_values("n_nationkey").reset_index(drop=True)
    nations_unique["nation_id"] = cudf.Series(range(len(nations_unique)), dtype="int64")

    final_df = final_df.merge(nations_unique[["n_nationkey", "nation_id"]], on="n_nationkey")

    num_nations = int(len(nations_unique))

    nation_ids_t, revenue_t = _query_core(
        torch.as_tensor(final_df["nation_id"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(final_df["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(final_df["l_discount"].to_cupy(), dtype=torch.float64),
        num_nations,
    )

    # Map back nation_id to n_name
    result_ids = cudf.Series(nation_ids_t, dtype="int64")
    result_rev = cudf.Series(revenue_t, dtype="float64")
    result = cudf.DataFrame({"nation_id": result_ids, "revenue": result_rev})
    result = result.merge(nations_unique[["nation_id", "n_name"]], on="nation_id")
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame({
        "n_name": result["n_name"],
        "revenue": result["revenue"],
    })
