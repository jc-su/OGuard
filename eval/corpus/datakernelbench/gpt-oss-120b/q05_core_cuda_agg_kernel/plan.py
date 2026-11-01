import os
import tempfile

import pandas as pd
import torch
import cudf
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA kernel source – compiled at import time.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ------------------------------------------------------------------
// Kernel: per‑row revenue aggregation into per‑nation buffer.
// ------------------------------------------------------------------
__global__ void agg_kernel(
    const int64_t* __restrict__ nation_id,
    const double*  __restrict__ extprice,
    const double*  __restrict__ discount,
    double*        __restrict__ out,
    int64_t N)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t nid   = nation_id[idx];
        double rev    = extprice[idx] * (1.0 - discount[idx]);
        // One atomic per row is acceptable – only ~25 distinct keys.
        atomicAdd(&out[nid], rev);
    }
}

// ------------------------------------------------------------------
// Launcher: prepares tensors, launches kernel, returns result.
// ------------------------------------------------------------------
torch::Tensor agg_launcher(
    torch::Tensor nation_id,
    torch::Tensor extprice,
    torch::Tensor discount,
    int64_t      num_nations)
{
    TORCH_CHECK(nation_id.dtype()   == torch::kInt64,
                "nation_id must be int64");
    TORCH_CHECK(extprice.dtype()   == torch::kFloat64,
                "extprice must be float64");
    TORCH_CHECK(discount.dtype()   == torch::kFloat64,
                "discount must be float64");

    // Ensure CUDA and contiguous layout.
    nation_id = nation_id.to(torch::kCUDA).contiguous();
    extprice  = extprice .to(torch::kCUDA).contiguous();
    discount  = discount .to(torch::kCUDA).contiguous();

    // Output buffer: per‑nation revenue, double‑precision.
    auto options = extprice.options();
    torch::Tensor out = torch::zeros({num_nations}, options);

    int64_t N = nation_id.numel();
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    agg_kernel<<<blocks, BLOCK_SIZE>>>(
        nation_id.data_ptr<int64_t>(),
        extprice.data_ptr<double>(),
        discount.data_ptr<double>(),
        out.data_ptr<double>(),
        N);

    // Propagate launch errors, if any.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
    return out;
}

// ------------------------------------------------------------------
// PyBind binding.
// ------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("agg_launcher", &agg_launcher,
          "Aggregate revenue per nation (CUDA, double precision)");
}
'''

def _load_cuda():
    """Compile the CUDA source into a Python extension and cache it."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q5_cuda_")
    cu_path = os.path.join(tmpdir, "q5_kernel.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="tpch_q5_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load (or re‑load) the compiled extension once per interpreter session.
_EXT = _load_cuda()

# ----------------------------------------------------------------------
# Private helper – launches the fused CUDA aggregation.
# ----------------------------------------------------------------------
def _launch_agg(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> torch.Tensor:
    """
    GPU‑accelerated aggregation: sum(l_extendedprice * (1‑discount))
    per nation_id.
    """
    # The launcher returns a 1‑D tensor of length `num_nations`.
    out = _EXT.agg_launcher(
        nation_id,
        l_extendedprice,
        l_discount,
        num_nations,
    )
    return out

# ----------------------------------------------------------------------
# Public core – unchanged signature, now uses the CUDA kernel.
# ----------------------------------------------------------------------
def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per‑nation revenue (double precision) via a fused CUDA kernel,
    then sort descending. Returns (ordered nation_ids, ordered revenues).
    """
    # Aggregation on GPU
    revenue = _launch_agg(nation_id, l_extendedprice, l_discount, num_nations)

    # Sorting – cheap, performed on the GPU tensor.
    nation_ids = torch.arange(num_nations, dtype=torch.int64, device=revenue.device)
    order = torch.argsort(revenue, descending=True)
    return nation_ids[order], revenue[order]

# ----------------------------------------------------------------------
# Run‑query – verbatim from the reference plan (joins, filters, etc.).
# ----------------------------------------------------------------------
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
    Description: TPC‑H Q5 — Local supplier volume for a region over one year.
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
    nation_f = df_nation.merge(
        region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey"
    )

    # Orders filtered by date
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (
        (orders_f["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
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
        left_on="o_orderkey",
        right_on="l_orderkey",
    )

    # Join with supplier on suppkey AND c_nationkey = s_nationkey
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    # Join with filtered nation on s_nationkey = n_nationkey
    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    # Build nation_id mapping from distinct nations in filter
    nations_unique = (
        nation_f[["n_nationkey", "n_name"]]
        .drop_duplicates()
        .sort_values("n_nationkey")
        .reset_index(drop=True)
    )
    nations_unique["nation_id"] = cudf.Series(
        range(len(nations_unique)), dtype="int64"
    )

    final_df = final_df.merge(
        nations_unique[["n_nationkey", "nation_id"]], on="n_nationkey"
    )

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
    result = result.merge(
        nations_unique[["nation_id", "n_name"]], on="nation_id"
    )
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return cudf.DataFrame(
        {"n_name": result["n_name"], "revenue": result["revenue"]}
    )
