import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: simple element‑wise kernel that computes the profit amount.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// ------------------------------------------------------------------
// Compute amount = price * (1 - discount) - supplycost * quantity
// ------------------------------------------------------------------
__global__ void amount_kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    double* __restrict__ out,
    const int64_t n)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    out[idx] = price[idx] * (1.0 - discount[idx]) - supplycost[idx] * quantity[idx];
}

// ------------------------------------------------------------------
// Launcher that validates inputs, allocates output, launches the kernel
// and returns the result tensor.
// ------------------------------------------------------------------
torch::Tensor amount_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor supplycost,
    torch::Tensor quantity)
{
    TORCH_CHECK(price.is_cuda() && discount.is_cuda()
                && supplycost.is_cuda() && quantity.is_cuda(),
                "All inputs must be CUDA tensors");

    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64, "supplycost must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");

    price = price.contiguous();
    discount = discount.contiguous();
    supplycost = supplycost.contiguous();
    quantity = quantity.contiguous();

    const int64_t n = price.numel();
    auto opts = price.options();
    torch::Tensor out = torch::empty({n}, opts);

    const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    amount_kernel<<<blocks, BLOCK_SIZE>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        supplycost.data_ptr<double>(),
        quantity.data_ptr<double>(),
        out.data_ptr<double>(),
        n);

    cudaDeviceSynchronize();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("amount_launcher", &amount_launcher, "Compute profit amount (CUDA)");
}
'''

def _load_cuda():
    """Compile the CUDA extension once and reuse it across imports."""
    # Fixed build location so subsequent imports reuse the compiled library.
    cache_dir = os.path.join(tempfile.gettempdir(), "tpch_q9_cuda_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cu_path = os.path.join(cache_dir, "q9_amount.cu")
    if not os.path.exists(cu_path):
        with open(cu_path, "w") as f:
            f.write(_CUDA_SOURCE)

    # The `load` function will rebuild only if the source has changed.
    return load(
        name="tpch_q9_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=cache_dir,
        verbose=False,
    )

# Load (or reuse) the compiled extension.
_EXT = _load_cuda()

def _launch_amount(
    price: torch.Tensor,
    discount: torch.Tensor,
    supplycost: torch.Tensor,
    quantity: torch.Tensor,
) -> torch.Tensor:
    """Execute the element‑wise amount kernel."""
    return _EXT.amount_launcher(price, discount, supplycost, quantity)

def run_query(
    tables: dict[str, cudf.DataFrame],
    name_pattern: str = "green",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT nation, o_year, sum(amount) AS sum_profit
        FROM (
            SELECT n_name AS nation,
                   EXTRACT(YEAR FROM o_orderdate) AS o_year,
                   l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
            FROM part, supplier, lineitem, partsupp, orders, nation
            WHERE s_suppkey = l_suppkey
              AND ps_suppkey = l_suppkey
              AND ps_partkey = l_partkey
              AND p_partkey = l_partkey
              AND o_orderkey = l_orderkey
              AND s_nationkey = n_nationkey
              AND p_name LIKE '%green%')
        AS profit
        GROUP BY nation, o_year
        ORDER BY nation ASC, o_year DESC;
    Description:
        All joins are performed with cuDF (GPU‑accelerated dataframe). After the
        final join we compute the profit amount with a single fused CUDA kernel,
        avoiding an intermediate Python/Torch element‑wise pass. The resulting
        amount column is then aggregated with cuDF’s highly optimized groupby,
        yielding a faster end‑to‑end execution on H100 while preserving exact
        semantics.
    Params:
        tables (dict[str, cudf.DataFrame]): mapping of table name to DataFrame.
        name_pattern (str): substring (without wildcards) to match part names.
    Returns:
        cudf.DataFrame with columns ['nation', 'o_year', 'sum_profit'].
    """
    # ------------------------------------------------------------------
    # 1) Load tables
    # ------------------------------------------------------------------
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_partsupp = tables["partsupp"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # ------------------------------------------------------------------
    # 2) Filter parts by name pattern
    # ------------------------------------------------------------------
    pat = name_pattern.replace("%", "")
    parts_filtered = df_part[df_part["p_name"].str.contains(pat)][["p_partkey"]]

    # ------------------------------------------------------------------
    # 3) Perform join chain, keeping only needed columns
    # ------------------------------------------------------------------
    li_parts = df_lineitem.merge(
        parts_filtered, left_on="l_partkey", right_on="p_partkey"
    )

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

    # ------------------------------------------------------------------
    # 4) Materialise necessary GPU tensors
    # ------------------------------------------------------------------
    price_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    supplycost_t = torch.as_tensor(joined["ps_supplycost"].to_cupy(), dtype=torch.float64)
    quantity_t = torch.as_tensor(joined["l_quantity"].to_cupy(), dtype=torch.float64)

    # ------------------------------------------------------------------
    # 5) Compute amount via fused CUDA kernel
    # ------------------------------------------------------------------
    amount_t = _launch_amount(price_t, discount_t, supplycost_t, quantity_t)

    # ------------------------------------------------------------------
    # 6) Build profit sub‑query DataFrame (cuDF)
    # ------------------------------------------------------------------
    profit_df = cudf.DataFrame({
        "nation": joined["n_name"].reset_index(drop=True),
        "o_year": joined["o_orderdate"].dt.year.astype("int32").reset_index(drop=True),
        "amount": cudf.Series(amount_t),
    })

    # ------------------------------------------------------------------
    # 7) Group by and aggregate
    # ------------------------------------------------------------------
    grouped = profit_df.groupby(["nation", "o_year"], as_index=False).agg({"amount": "sum"})
    grouped = grouped.rename(columns={"amount": "sum_profit"})
    grouped = grouped.sort_values(["nation", "o_year"], ascending=[True, False]).reset_index(drop=True)

    return cudf.DataFrame({
        "nation": grouped["nation"],
        "o_year": grouped["o_year"],
        "sum_profit": grouped["sum_profit"],
    })
