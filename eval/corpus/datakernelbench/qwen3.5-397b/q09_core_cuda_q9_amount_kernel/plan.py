import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void q9_amount_kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    double* __restrict__ amount,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double p = price[idx];
        double d = discount[idx];
        double s = supplycost[idx];
        double q = quantity[idx];
        amount[idx] = p * (1.0 - d) - s * q;
    }
}

torch::Tensor q9_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor supplycost,
    torch::Tensor quantity
) {
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64, "supplycost must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");

    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!supplycost.is_cuda()) supplycost = supplycost.cuda();
    if (!quantity.is_cuda()) quantity = quantity.cuda();

    price = price.contiguous();
    discount = discount.contiguous();
    supplycost = supplycost.contiguous();
    quantity = quantity.contiguous();

    int64_t n = price.numel();
    auto amount = torch::empty({n}, price.options());

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q9_amount_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        supplycost.data_ptr<double>(),
        quantity.data_ptr<double>(),
        amount.data_ptr<double>(),
        n
    );

    return amount;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q9_launcher", &q9_launcher, "TPC-H Q9 amount calculation (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q9_cuda_")
    cu_path = os.path.join(tmpdir, "_kernels.cu")
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

def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    return _EXT.q9_launcher(
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
