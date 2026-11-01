import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void q9_kernel(
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const double* __restrict__ ps_supplycost,
    const double* __restrict__ l_quantity,
    double* __restrict__ amount_out,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        amount_out[idx] = l_extendedprice[idx] * (1.0 - l_discount[idx]) - ps_supplycost[idx] * l_quantity[idx];
    }
}

torch::Tensor q9_launcher(
    torch::Tensor l_extendedprice_tensor,
    torch::Tensor l_discount_tensor,
    torch::Tensor ps_supplycost_tensor,
    torch::Tensor l_quantity_tensor
) {
    TORCH_CHECK(l_extendedprice_tensor.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_discount_tensor.scalar_type() == torch::kFloat64);
    TORCH_CHECK(ps_supplycost_tensor.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_quantity_tensor.scalar_type() == torch::kFloat64);

    auto l_extendedprice_contig = l_extendedprice_tensor.contiguous();
    auto l_discount_contig = l_discount_tensor.contiguous();
    auto ps_supplycost_contig = ps_supplycost_tensor.contiguous();
    auto l_quantity_contig = l_quantity_tensor.contiguous();

    if (!l_extendedprice_contig.is_cuda()) l_extendedprice_contig = l_extendedprice_contig.cuda();
    if (!l_discount_contig.is_cuda()) l_discount_contig = l_discount_contig.cuda();
    if (!ps_supplycost_contig.is_cuda()) ps_supplycost_contig = ps_supplycost_contig.cuda();
    if (!l_quantity_contig.is_cuda()) l_quantity_contig = l_quantity_contig.cuda();

    auto amount_out = torch::empty_like(l_extendedprice_contig);
    int64_t n = l_extendedprice_contig.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q9_kernel<<<blocks, BLOCK_SIZE>>>(
        l_extendedprice_contig.data_ptr<double>(),
        l_discount_contig.data_ptr<double>(),
        ps_supplycost_contig.data_ptr<double>(),
        l_quantity_contig.data_ptr<double>(),
        amount_out.data_ptr<double>(),
        n
    );
    return amount_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q9_launcher", &q9_launcher, "TPC-H Q9 amount (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q9_cuda_")
    cu_path = os.path.join(tmpdir, "_q9_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q9_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _launch_q9(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    return _EXT.q9_launcher(l_extendedprice, l_discount, ps_supplycost, l_quantity)

def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    return _launch_q9(l_extendedprice, l_discount, ps_supplycost, l_quantity)

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
        "amount": cudf.Series(amount_t.cpu().numpy()),
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
