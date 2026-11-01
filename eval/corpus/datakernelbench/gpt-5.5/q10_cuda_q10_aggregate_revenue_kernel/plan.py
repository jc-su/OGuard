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
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void q10_aggregate_revenue_kernel(
    const int64_t* __restrict__ custkey_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ out_revenue_ptr,
    int64_t n_rows,
    int64_t out_size
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_rows) {
        int64_t ck = custkey_ptr[idx];
        if (ck >= 0 && ck < out_size) {
            double rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            atomicAdd(out_revenue_ptr + ck, rev);
        }
    }
}

torch::Tensor q10_aggregate_revenue_launcher(
    torch::Tensor custkey,
    torch::Tensor price,
    torch::Tensor discount,
    int64_t out_size
) {
    TORCH_CHECK(out_size >= 0, "out_size must be non-negative");
    TORCH_CHECK(custkey.scalar_type() == torch::kInt64, "custkey must be int64");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(custkey.numel() == price.numel(), "custkey and price length mismatch");
    TORCH_CHECK(price.numel() == discount.numel(), "price and discount length mismatch");

    if (!custkey.is_cuda()) custkey = custkey.to(torch::kCUDA);
    if (!price.is_cuda()) price = price.to(torch::kCUDA);
    if (!discount.is_cuda()) discount = discount.to(torch::kCUDA);

    const c10::cuda::CUDAGuard device_guard(price.device());

    custkey = custkey.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();

    auto out = torch::zeros({out_size}, price.options().dtype(torch::kFloat64));
    int64_t n_rows = price.numel();
    if (n_rows == 0 || out_size == 0) {
        return out;
    }

    int blocks = static_cast<int>((n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    q10_aggregate_revenue_kernel<<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
        custkey.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        out.data_ptr<double>(),
        n_rows,
        out_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q10_aggregate_revenue_launcher", &q10_aggregate_revenue_launcher,
          "TPC-H Q10 revenue aggregation by customer key (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile the CUDA extension once at import; device code lives in _CUDA_SOURCE."""
    tmpdir = tempfile.mkdtemp(prefix="q10_cuda_")
    cu_path = os.path.join(tmpdir, "_q10_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q10_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _empty_result(df_customer: cudf.DataFrame, df_nation: cudf.DataFrame) -> cudf.DataFrame:
    """Build an empty cuDF result with the exact Q10 output columns and dtypes."""
    result = cudf.DataFrame()
    result["c_custkey"] = df_customer["c_custkey"].head(0)
    result["c_name"] = df_customer["c_name"].head(0)
    result["revenue"] = cudf.Series([], dtype="float64")
    result["c_acctbal"] = df_customer["c_acctbal"].head(0)
    result["n_name"] = df_nation["n_name"].head(0)
    result["c_address"] = df_customer["c_address"].head(0)
    result["c_phone"] = df_customer["c_phone"].head(0)
    result["c_comment"] = df_customer["c_comment"].head(0)
    return result


def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    revenue_vector_size: int,
) -> torch.Tensor:
    """
    Run the numeric GROUP BY hot path on the GPU in one CUDA kernel.

    The input rows are the cuDF join result for returned lineitems matched to dated
    orders. CUDA computes l_extendedprice * (1 - l_discount) and atomically
    accumulates directly into a dense revenue-by-customer vector. This avoids the
    reference Torch unique/inverse/scatter pipeline and its temporary tensors.
    """
    return _EXT.q10_aggregate_revenue_launcher(
        c_custkey,
        l_extendedprice,
        l_discount,
        int(revenue_vector_size),
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    orderdate_lo: str = "1993-10-01",
    orderdate_hi: str = "1994-01-01",
    returnflag_val: str = "R",
    top_n: int = 20,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT c_custkey, c_name,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               c_acctbal, n_name, c_address, c_phone, c_comment
        FROM customer, orders, lineitem, nation
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate >= CAST('1993-10-01' AS date)
          AND o_orderdate < CAST('1994-01-01' AS date)
          AND l_returnflag = 'R'
          AND c_nationkey = n_nationkey
        GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
        ORDER BY revenue DESC
        LIMIT 20;

    Description:
        TPC-H Q10 returned item reporting. cuDF performs date/string filtering,
        large relational joins, and final string/dimension column assembly. The
        numeric revenue aggregation runs in a hand-written CUDA kernel: joined
        rows are scanned once, revenue is projected in registers, and values are
        accumulated into a dense customer-key revenue vector. This replaces the
        reference Torch unique + inverse + scatter_add group-by hot path with
        fewer temporary tensors and a single fused GPU aggregation pass.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables keyed by table name.
        orderdate_lo (str): Inclusive lower bound for o_orderdate.
        orderdate_hi (str): Exclusive upper bound for o_orderdate.
        returnflag_val (str): Required l_returnflag value.
        top_n (int): Number of top revenue customers to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_nation = tables["nation"]

    n_top = int(top_n)
    if n_top <= 0:
        return _empty_result(df_customer, df_nation)

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    lo_join = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[
        ["o_custkey", "l_extendedprice", "l_discount"]
    ]

    if len(lo_join) == 0:
        return _empty_result(df_customer, df_nation)

    max_customer_key = int(df_customer["c_custkey"].max())
    revenue_vector_size = max_customer_key + 1

    custkey_t = torch.as_tensor(lo_join["o_custkey"].to_cupy(), dtype=torch.int64)
    price_t = torch.as_tensor(lo_join["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(lo_join["l_discount"].to_cupy(), dtype=torch.float64)

    revenue_by_key = _query_core(custkey_t, price_t, discount_t, revenue_vector_size)

    k = min(n_top, int(revenue_by_key.numel()))
    if k <= 0:
        return _empty_result(df_customer, df_nation)

    top_rev, top_keys = torch.topk(revenue_by_key, k)
    positive_mask = top_rev > 0.0
    top_rev = top_rev[positive_mask]
    top_keys = top_keys[positive_mask].to(torch.int64)

    if int(top_rev.numel()) == 0:
        return _empty_result(df_customer, df_nation)

    agg_df = cudf.DataFrame(
        {
            "c_custkey": top_keys,
            "revenue": top_rev,
        }
    )

    cust_cols = df_customer[
        ["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]
    ]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")
    agg_df = agg_df.merge(
        df_nation[["n_nationkey", "n_name"]],
        left_on="c_nationkey",
        right_on="n_nationkey",
    )
    agg_df = agg_df.sort_values("revenue", ascending=False).head(n_top)

    result = cudf.DataFrame(
        {
            "c_custkey": agg_df["c_custkey"].reset_index(drop=True),
            "c_name": agg_df["c_name"].reset_index(drop=True),
            "revenue": agg_df["revenue"].reset_index(drop=True),
            "c_acctbal": agg_df["c_acctbal"].reset_index(drop=True),
            "n_name": agg_df["n_name"].reset_index(drop=True),
            "c_address": agg_df["c_address"].reset_index(drop=True),
            "c_phone": agg_df["c_phone"].reset_index(drop=True),
            "c_comment": agg_df["c_comment"].reset_index(drop=True),
        }
    )
    return result
