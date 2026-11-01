import os
import tempfile
import torch
import cudf
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA device code (compiled at import time)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// Element‑wise computation of profit amount:
// amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity
__global__ void compute_amount_kernel(
    const double* __restrict__ extprice,
    const double* __restrict__ discount,
    const double* __restrict__ supplycost,
    const double* __restrict__ quantity,
    double* __restrict__ out,
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double ep = extprice[idx];
        double d  = discount[idx];
        double sc = supplycost[idx];
        double q  = quantity[idx];
        out[idx] = ep * (1.0 - d) - sc * q;
    }
}

// Launcher returns a brand‑new tensor containing the element‑wise results.
torch::Tensor compute_amount_launcher(
    torch::Tensor extprice,
    torch::Tensor discount,
    torch::Tensor supplycost,
    torch::Tensor quantity)
{
    TORCH_CHECK(extprice.scalar_type() == torch::kFloat64,
                "l_extendedprice must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64,
                "l_discount must be float64");
    TORCH_CHECK(supplycost.scalar_type() == torch::kFloat64,
                "ps_supplycost must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64,
                "l_quantity must be float64");

    // Ensure CUDA and contiguity.
    if (!extprice.is_cuda())     extprice = extprice.cuda();
    if (!discount.is_cuda())     discount = discount.cuda();
    if (!supplycost.is_cuda())   supplycost = supplycost.cuda();
    if (!quantity.is_cuda())     quantity = quantity.cuda();

    extprice   = extprice.contiguous();
    discount   = discount.contiguous();
    supplycost = supplycost.contiguous();
    quantity   = quantity.contiguous();

    int64_t N = extprice.numel();
    auto out = torch::empty_like(extprice);

    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>((N + threads - 1) / threads);

    compute_amount_kernel<<<blocks, threads>>>(
        extprice.data_ptr<double>(),
        discount.data_ptr<double>(),
        supplycost.data_ptr<double>(),
        quantity.data_ptr<double>(),
        out.data_ptr<double>(),
        N);

    // Propagate any launch errors.
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel launch failed");

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_amount_launcher", &compute_amount_launcher,
          "Element‑wise profit amount (CUDA, double precision)");
}
'''

def _load_cuda():
    """Compile the CUDA source into a PyTorch extension and return the module."""
    tmpdir = tempfile.mkdtemp(prefix="q9_cuda_")
    cu_path = os.path.join(tmpdir, "q9_kernels.cu")
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
        verbose=False,
    )

# Load (or rebuild) the compiled extension at import time.
_EXT = _load_cuda()

# ----------------------------------------------------------------------
# Private helper that invokes the compiled kernel
# ----------------------------------------------------------------------
def _launch_compute_amount(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Launch the fused CUDA kernel that computes the profit amount per row.
    All tensors must be 1‑D, float64, GPU resident.
    Returns a new GPU tensor of the same length.
    """
    return _EXT.compute_amount_launcher(
        l_extendedprice,
        l_discount,
        ps_supplycost,
        l_quantity,
    )


# ----------------------------------------------------------------------
# Public compute core (level: core) – signature fixed by the task
# ----------------------------------------------------------------------
def _query_core(
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    ps_supplycost: torch.Tensor,
    l_quantity: torch.Tensor,
) -> torch.Tensor:
    """
    Compute element‑wise amount:
        amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity

    This implementation offloads the element‑wise work to a single fused
    CUDA kernel, removing Python‑level element loops and temporary tensors.
    The kernel runs in double precision (float64) to match the reference.
    """
    return _launch_compute_amount(
        l_extendedprice,
        l_discount,
        ps_supplycost,
        l_quantity,
    )


# ----------------------------------------------------------------------
# Unchanged query orchestration – copied verbatim from the reference plan
# ----------------------------------------------------------------------
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

    # Compute amount via the fused CUDA kernel
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
