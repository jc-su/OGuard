"""TPC-H Q9 core CUDA optimization module.

The relational work (cuDF string filter, joins, groupby, sort, and final frame
construction) remains exactly in ``run_query``.  The numeric profit expression
runs on the GPU in a hand-written CUDA extension as one fused coalesced
elementwise pass, avoiding the multiple temporary tensors and eager PyTorch
kernel launches used by the reference hot path.
"""

import os
import tempfile
from typing import Any

import torch
import cudf
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

#define BLOCK_SIZE 256

__global__ void q9_amount_kernel(
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const double* __restrict__ ps_supplycost,
    const double* __restrict__ l_quantity,
    double* __restrict__ out,
    int64_t n_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double discounted_price = l_extendedprice[idx] * (1.0 - l_discount[idx]);
        double supply_cost = ps_supplycost[idx] * l_quantity[idx];
        out[idx] = discounted_price - supply_cost;
    }
}

torch::Tensor q9_amount_launcher(
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    torch::Tensor ps_supplycost,
    torch::Tensor l_quantity
) {
    TORCH_CHECK(l_extendedprice.is_cuda(), "l_extendedprice must be a CUDA tensor");
    TORCH_CHECK(l_discount.is_cuda(), "l_discount must be a CUDA tensor");
    TORCH_CHECK(ps_supplycost.is_cuda(), "ps_supplycost must be a CUDA tensor");
    TORCH_CHECK(l_quantity.is_cuda(), "l_quantity must be a CUDA tensor");

    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    TORCH_CHECK(ps_supplycost.scalar_type() == torch::kFloat64, "ps_supplycost must be float64");
    TORCH_CHECK(l_quantity.scalar_type() == torch::kFloat64, "l_quantity must be float64");

    TORCH_CHECK(l_extendedprice.dim() == 1, "l_extendedprice must be 1D");
    TORCH_CHECK(l_discount.dim() == 1, "l_discount must be 1D");
    TORCH_CHECK(ps_supplycost.dim() == 1, "ps_supplycost must be 1D");
    TORCH_CHECK(l_quantity.dim() == 1, "l_quantity must be 1D");

    const int64_t n = l_extendedprice.numel();
    TORCH_CHECK(l_discount.numel() == n, "l_discount length mismatch");
    TORCH_CHECK(ps_supplycost.numel() == n, "ps_supplycost length mismatch");
    TORCH_CHECK(l_quantity.numel() == n, "l_quantity length mismatch");

    TORCH_CHECK(l_discount.device() == l_extendedprice.device(), "l_discount device mismatch");
    TORCH_CHECK(ps_supplycost.device() == l_extendedprice.device(), "ps_supplycost device mismatch");
    TORCH_CHECK(l_quantity.device() == l_extendedprice.device(), "l_quantity device mismatch");

    c10::cuda::CUDAGuard device_guard(l_extendedprice.device());

    torch::Tensor price_c = l_extendedprice.contiguous();
    torch::Tensor discount_c = l_discount.contiguous();
    torch::Tensor supplycost_c = ps_supplycost.contiguous();
    torch::Tensor quantity_c = l_quantity.contiguous();

    torch::Tensor out = torch::empty({n}, price_c.options());
    if (n == 0) {
        return out;
    }

    const int64_t blocks64 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    TORCH_CHECK(blocks64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                "too many CUDA blocks requested");
    const int blocks = static_cast<int>(blocks64);

    q9_amount_kernel<<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
        price_c.data_ptr<double>(),
        discount_c.data_ptr<double>(),
        supplycost_c.data_ptr<double>(),
        quantity_c.data_ptr<double>(),
        out.data_ptr<double>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q9_amount_launcher", &q9_amount_launcher,
          "TPC-H Q9 fused amount expression (CUDA)");
}
'''


def _load_cuda() -> Any:
    """Compile and load the CUDA extension from an embedded .cu source string.

    The extension is built once at module import.  Device code lives only in
    ``_CUDA_SOURCE`` and is compiled for H100 (sm_90).  The resulting launcher
    performs the Q9 amount projection as a single memory-bound GPU pass.
    """
    tmpdir = tempfile.mkdtemp(prefix="q9_cuda_core_")
    cu_path = os.path.join(tmpdir, "_q9_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w", encoding="utf-8") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q9_cuda_core_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90", "--fmad=false"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """Compute Q9 row profit amounts on the GPU using a fused CUDA kernel.

    Inputs are 1D float64 CUDA tensors extracted from the cuDF join result.
    The CUDA kernel performs coalesced loads and computes
    ``l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity`` in one
    launch, writing the output tensor directly.  This is faster than eager
    PyTorch for this memory-bound projection because it avoids several
    intermediate tensors and multiple kernel launches.
    """
    return _EXT.q9_amount_launcher(
        l_extendedprice,
        l_discount,
        ps_supplycost,
        l_quantity,
    )


def run_query(tables: dict[str, cudf.DataFrame], name_pattern: str = "green") -> cudf.DataFrame:
    """
    SQL: SELECT nation, o_year, sum(amount) AS sum_profit FROM (SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount FROM part, supplier, lineitem, partsupp, orders, nation WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey AND p_name LIKE '%green%') AS profit GROUP BY nation, o_year ORDER BY nation, o_year DESC;
    Description: TPC-H Q9 - Product Type Profit Measure. Computes profit by nation and year for parts whose name matches a pattern.
    Params: name_pattern (str): substring to filter part names (without % wildcards).
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    pat = name_pattern.replace("%", "")
    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]

    li_parts = df_lineitem.merge(parts_filtered, left_on="l_partkey", right_on="p_partkey")

    li_ps = li_parts.merge(
        df_partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]],
        left_on=["l_partkey", "l_suppkey"],
        right_on=["ps_partkey", "ps_suppkey"],
    )

    li_ps_s = li_ps.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )

    li_ps_s_o = li_ps_s.merge(
        df_orders[["o_orderkey", "o_orderdate"]],
        left_on="l_orderkey",
        right_on="o_orderkey",
    )

    joined = li_ps_s_o.merge(
        df_nation[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    # Compute amount via torch core
    amount_t = _query_core(
        torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64),
    )

    # Build profit subquery frame in cuDF
    profit_df = cudf.DataFrame({
        "nation": joined["n_name"].reset_index(drop=True),
        "o_year": joined["o_orderdate"].dt.year.astype("int32").reset_index(drop=True),
        "amount": cudf.Series(amount_t),
    })

    # Group by nation, o_year and sum
    grouped = profit_df.groupby(["nation", "o_year"], as_index=False).agg({"amount": "sum"})
    grouped = grouped.rename(columns={"amount": "sum_profit"})

    # Order by nation ASC, o_year DESC
    grouped = grouped.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)

    return cudf.DataFrame({
        "nation": grouped["nation"],
        "o_year": grouped["o_year"],
        "sum_profit": grouped["sum_profit"],
    })
