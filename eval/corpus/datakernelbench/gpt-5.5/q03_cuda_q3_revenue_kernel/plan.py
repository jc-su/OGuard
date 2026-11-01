from __future__ import annotations

import os
import tempfile
from typing import Any

import cupy as cp
import cudf
import numpy as np
import pandas as pd
import torch
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void q3_revenue_kernel(
    const double* __restrict__ extendedprice_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ revenue_ptr,
    int64_t n_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        revenue_ptr[idx] = extendedprice_ptr[idx] * (1.0 - discount_ptr[idx]);
    }
}

torch::Tensor q3_revenue_launcher(
    torch::Tensor extendedprice,
    torch::Tensor discount
) {
    TORCH_CHECK(extendedprice.is_cuda(), "extendedprice must be CUDA");
    TORCH_CHECK(discount.is_cuda(), "discount must be CUDA");
    TORCH_CHECK(extendedprice.scalar_type() == torch::kFloat64, "extendedprice must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(extendedprice.numel() == discount.numel(), "inputs must have the same length");
    TORCH_CHECK(extendedprice.get_device() == discount.get_device(), "inputs must be on the same device");

    extendedprice = extendedprice.contiguous();
    discount = discount.contiguous();

    int64_t n = extendedprice.numel();
    auto revenue = torch::empty({n}, extendedprice.options());

    if (n > 0) {
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        q3_revenue_kernel<<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            extendedprice.data_ptr<double>(),
            discount.data_ptr<double>(),
            revenue.data_ptr<double>(),
            n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return revenue;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q3_revenue_launcher", &q3_revenue_launcher, "TPC-H Q3 revenue projection CUDA kernel");
}
'''


def _load_cuda() -> Any:
    """
    Build the CUDA extension from the embedded source string.

    The compiled GPU kernel performs the memory-bound Q3 revenue projection in
    one coalesced pass over two float64 columns, avoiding PyTorch eager
    expression temporaries while cuDF handles joins and hash aggregation.
    """
    tmpdir = tempfile.mkdtemp(prefix="q3_cuda_fast_")
    cu_path = os.path.join(tmpdir, "_q3_cuda.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q3_cuda_fast_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT: Any = _load_cuda()


def _empty_result() -> cudf.DataFrame:
    """Create an empty result with the exact output column order and dtypes."""
    return cudf.DataFrame(
        {
            "l_orderkey": cudf.Series([], dtype="int64"),
            "revenue": cudf.Series([], dtype="float64"),
            "o_orderdate": cudf.Series([], dtype="datetime64[s]"),
            "o_shippriority": cudf.Series([], dtype="int32"),
        }
    )


def _series_to_torch_f64(series: cudf.Series) -> torch.Tensor:
    """
    Zero-copy hand off a cuDF numeric column to a CUDA torch.float64 tensor.

    The data stays on GPU through cuDF -> CuPy -> Torch interop; the CUDA
    extension consumes the resulting device pointer directly.
    """
    return torch.as_tensor(series.to_cupy(), dtype=torch.float64)


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> torch.Tensor:
    """
    Execute the row-wise revenue expression on the GPU using hand-written CUDA.

    This is the arithmetic hot path after cuDF has compacted the joined rows.
    It fuses `extendedprice * (1 - discount)` into a single CUDA launch with
    coalesced loads and one output write per row.
    """
    return _EXT.q3_revenue_launcher(l_extendedprice, l_discount)


def run_query(
    tables: dict[str, cudf.DataFrame],
    mktsegment: str = "BUILDING",
    orderdate_hi: str = "1995-03-15",
    shipdate_lo: str = "1995-03-15",
    top_n: int = 10,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate,
               o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < CAST('1995-03-15' AS date)
          AND l_shipdate > CAST('1995-03-15' AS date)
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;

    Description:
        TPC-H Q3. cuDF performs the string predicate, date predicate pushdown,
        GPU hash joins, GPU groupby, and final sort. The joined row-wise revenue
        projection runs in a custom CUDA kernel. This plan is faster than the
        reference because it pushes selective date predicates before joins,
        avoids Torch boolean-mask/unique/scatter aggregation on the large joined
        table, carries only required columns, and computes revenue in one fused
        GPU pass without intermediate PyTorch tensors.

    Params:
        tables (dict[str, cudf.DataFrame]): Input tables keyed by TPCH name.
        mktsegment (str): Customer market segment filter.
        orderdate_hi (str): Exclusive upper bound for o_orderdate.
        shipdate_lo (str): Exclusive lower bound for l_shipdate.
        top_n (int): Number of rows to return.
    """
    limit = int(top_n)
    if limit <= 0:
        return _empty_result()

    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    orderdate_hi_ts = np.datetime64(pd.to_datetime(orderdate_hi).to_datetime64(), "s")
    shipdate_lo_ts = np.datetime64(pd.to_datetime(shipdate_lo).to_datetime64(), "s")

    cust_f = df_customer.loc[
        df_customer["c_mktsegment"] == mktsegment,
        ["c_custkey"],
    ]
    if len(cust_f) == 0:
        return _empty_result()

    orders_f = df_orders.loc[
        df_orders["o_orderdate"] < orderdate_hi_ts,
        ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
    ]
    if len(orders_f) == 0:
        return _empty_result()

    orders_f = orders_f.copy(deep=False)
    orders_f["o_shippriority"] = orders_f["o_shippriority"].astype("int32")

    co = cust_f.merge(orders_f, left_on="c_custkey", right_on="o_custkey", how="inner")
    if len(co) == 0:
        return _empty_result()

    co = co[["o_orderkey", "o_orderdate", "o_shippriority"]]

    lineitem_f = df_lineitem.loc[
        df_lineitem["l_shipdate"] > shipdate_lo_ts,
        ["l_orderkey", "l_extendedprice", "l_discount"],
    ]
    if len(lineitem_f) == 0:
        return _empty_result()

    lineitem_f = lineitem_f.rename(columns={"l_orderkey": "o_orderkey"})
    joined = co.merge(lineitem_f, on="o_orderkey", how="inner")
    if len(joined) == 0:
        return _empty_result()

    revenue_t = _query_core(
        _series_to_torch_f64(joined["l_extendedprice"]),
        _series_to_torch_f64(joined["l_discount"]),
    )
    joined["revenue"] = cudf.Series(cp.from_dlpack(revenue_t), index=joined.index)

    grouped = (
        joined.groupby(["o_orderkey", "o_orderdate", "o_shippriority"], sort=False)["revenue"]
        .sum()
        .reset_index()
    )
    if len(grouped) == 0:
        return _empty_result()

    ordered = grouped.sort_values(
        by=["revenue", "o_orderdate"],
        ascending=[False, True],
    ).head(limit)

    out = cudf.DataFrame()
    out["l_orderkey"] = ordered["o_orderkey"]
    out["revenue"] = ordered["revenue"]
    out["o_orderdate"] = ordered["o_orderdate"]
    out["o_shippriority"] = ordered["o_shippriority"].astype("int32")
    return out[["l_orderkey", "revenue", "o_orderdate", "o_shippriority"]].reset_index(drop=True)
