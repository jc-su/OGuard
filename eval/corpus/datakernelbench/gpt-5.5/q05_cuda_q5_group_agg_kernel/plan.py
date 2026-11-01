import os
import tempfile
from typing import Any

import cudf
import pandas as pd
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define MAX_GROUPS 32

__global__ void q5_group_agg_kernel(
    const int64_t* __restrict__ nation_id_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    int64_t n_elements,
    int32_t num_nations,
    double* __restrict__ out_ptr
) {
    __shared__ double shared_accum[MAX_GROUPS];

    for (int i = threadIdx.x; i < num_nations; i += blockDim.x) {
        shared_accum[i] = 0.0;
    }
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t gid64 = nation_id_ptr[idx];
        if (gid64 >= 0 && gid64 < num_nations) {
            int gid = static_cast<int>(gid64);
            double revenue = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            atomicAdd(&shared_accum[gid], revenue);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_nations; i += blockDim.x) {
        double v = shared_accum[i];
        if (v != 0.0) {
            atomicAdd(&out_ptr[i], v);
        }
    }
}

torch::Tensor q5_aggregate_launcher(
    torch::Tensor nation_id,
    torch::Tensor price,
    torch::Tensor discount,
    int64_t num_nations64
) {
    TORCH_CHECK(num_nations64 >= 0, "num_nations must be non-negative");
    TORCH_CHECK(num_nations64 <= MAX_GROUPS, "q5 CUDA aggregator supports at most 32 groups");
    TORCH_CHECK(nation_id.scalar_type() == torch::kInt64, "nation_id must be int64");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(nation_id.numel() == price.numel(), "nation_id and price length mismatch");
    TORCH_CHECK(price.numel() == discount.numel(), "price and discount length mismatch");
    TORCH_CHECK(nation_id.is_cuda(), "nation_id must be a CUDA tensor");
    TORCH_CHECK(price.is_cuda(), "price must be a CUDA tensor");
    TORCH_CHECK(discount.is_cuda(), "discount must be a CUDA tensor");
    TORCH_CHECK(nation_id.device() == price.device(), "nation_id and price must be on same device");
    TORCH_CHECK(price.device() == discount.device(), "price and discount must be on same device");

    c10::cuda::CUDAGuard device_guard(price.device());

    nation_id = nation_id.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();

    int64_t n = price.numel();
    int32_t num_nations = static_cast<int32_t>(num_nations64);
    auto out = torch::zeros({num_nations64}, price.options().dtype(torch::kFloat64));

    if (n == 0 || num_nations == 0) {
        return out;
    }

    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    q5_group_agg_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        nation_id.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        n,
        num_nations,
        out.data_ptr<double>()
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "q5_group_agg_kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_aggregate_launcher", &q5_aggregate_launcher, "TPC-H Q5 grouped revenue aggregation (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the small Q5 CUDA extension once; device code performs the fused revenue projection and grouped aggregation."""
    tmpdir = tempfile.mkdtemp(prefix="q5_cuda_")
    cu_path = os.path.join(tmpdir, "_q5_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q5_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT: Any = _load_cuda()


def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> torch.Tensor:
    """
    Run the numeric hot path on the GPU with hand-written CUDA.

    The CUDA kernel fuses projection ``l_extendedprice * (1 - l_discount)`` with
    grouped aggregation by the compact nation id.  It uses one shared-memory
    accumulator array per block and emits only one global atomic add per
    non-empty group per block, avoiding PyTorch temporaries and scatter overhead.
    """
    return _EXT.q5_aggregate_launcher(
        nation_id,
        l_extendedprice,
        l_discount,
        int(num_nations),
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "ASIA",
    orderdate_lo: str = "1994-01-01",
    orderdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL:
      SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
      FROM customer, orders, lineitem, supplier, nation, region
      WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
        AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
        AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
        AND r_name = 'ASIA'
        AND o_orderdate >= DATE '1994-01-01'
        AND o_orderdate <  DATE '1995-01-01'
      GROUP BY n_name
      ORDER BY revenue DESC;

    Description:
      TPC-H Q5 — local supplier volume for a region over one year.  cuDF handles
      string filtering and relational joins, but the join order is rewritten to
      filter customers and suppliers to the selected region before touching the
      large lineitem join.  The final revenue projection and GROUP BY over the
      compact region nation ids runs in a fused CUDA kernel, reducing PyTorch
      eager temporaries and replacing scatter-add with block-local accumulation.

    Params:
      tables: Mapping containing cuDF DataFrames for customer, orders, lineitem,
        supplier, nation, and region.
      region_name: Region to filter on.
      orderdate_lo: Inclusive lower bound on order date.
      orderdate_hi: Exclusive upper bound on order date.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    region_f = df_region[df_region["r_name"] == region_name]
    nation_f = df_nation[["n_nationkey", "n_name", "n_regionkey"]].merge(
        region_f[["r_regionkey"]],
        left_on="n_regionkey",
        right_on="r_regionkey",
    )

    nations_unique = (
        nation_f[["n_nationkey", "n_name"]]
        .drop_duplicates()
        .sort_values("n_nationkey")
        .reset_index(drop=True)
    )
    num_nations = int(len(nations_unique))
    nations_unique["nation_id"] = cudf.Series(range(num_nations), dtype="int64")

    if num_nations == 0:
        return cudf.DataFrame(
            {
                "n_name": nations_unique["n_name"],
                "revenue": cudf.Series([], dtype="float64"),
            }
        )

    nation_map = nations_unique[["n_nationkey", "nation_id"]]

    customer_f = df_customer[["c_custkey", "c_nationkey"]].merge(
        nation_map,
        left_on="c_nationkey",
        right_on="n_nationkey",
    )
    customer_f = customer_f[["c_custkey", "c_nationkey", "nation_id"]]

    supplier_f = df_supplier[["s_suppkey", "s_nationkey"]].merge(
        nation_map[["n_nationkey"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )
    supplier_f = supplier_f[["s_suppkey", "s_nationkey"]]

    orderdate_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders.loc[
        (orderdate_days >= lo_days) & (orderdate_days < hi_days),
        ["o_orderkey", "o_custkey"],
    ]

    co = customer_f.merge(
        orders_f,
        left_on="c_custkey",
        right_on="o_custkey",
    )
    co = co[["o_orderkey", "c_nationkey", "nation_id"]]

    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey",
        right_on="l_orderkey",
    )

    final_df = col.merge(
        supplier_f,
        left_on=["l_suppkey", "c_nationkey"],
        right_on=["s_suppkey", "s_nationkey"],
    )

    nation_id_t = torch.as_tensor(final_df["nation_id"].to_cupy(), dtype=torch.int64)
    price_t = torch.as_tensor(final_df["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(final_df["l_discount"].to_cupy(), dtype=torch.float64)

    revenue_t = _query_core(nation_id_t, price_t, discount_t, num_nations)
    if revenue_t.is_cuda:
        torch.cuda.synchronize(revenue_t.device)

    result = cudf.DataFrame(
        {
            "nation_id": cudf.Series(range(num_nations), dtype="int64"),
            "revenue": cudf.Series(revenue_t, dtype="float64"),
        }
    )
    result = result.merge(nations_unique[["nation_id", "n_name"]], on="nation_id")
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame(
        {
            "n_name": result["n_name"],
            "revenue": result["revenue"],
        }
    )
