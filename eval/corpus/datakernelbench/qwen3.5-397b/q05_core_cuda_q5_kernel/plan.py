import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024

__global__ void q5_kernel(
    const int64_t* __restrict__ nation_id_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int64_t nid = nation_id_ptr[idx];
        double rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
        atomicAdd(&out_ptr[nid], rev);
    }
}

std::vector<at::Tensor> q5_launcher(
    at::Tensor nation_id,
    at::Tensor l_extendedprice,
    at::Tensor l_discount,
    int64_t num_nations
) {
    TORCH_CHECK(nation_id.scalar_type() == torch::kInt64, "nation_id must be int64");
    TORCH_CHECK(l_extendedprice.scalar_type() == torch::kFloat64, "l_extendedprice must be float64");
    TORCH_CHECK(l_discount.scalar_type() == torch::kFloat64, "l_discount must be float64");
    
    if (!nation_id.is_cuda()) nation_id = nation_id.cuda();
    if (!l_extendedprice.is_cuda()) l_extendedprice = l_extendedprice.cuda();
    if (!l_discount.is_cuda()) l_discount = l_discount.cuda();
    
    nation_id = nation_id.contiguous();
    l_extendedprice = l_extendedprice.contiguous();
    l_discount = l_discount.contiguous();
    
    int64_t n = nation_id.numel();
    auto out = torch::zeros({num_nations}, l_extendedprice.options());
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q5_kernel<<<blocks, BLOCK_SIZE>>>(
        nation_id.data_ptr<int64_t>(),
        l_extendedprice.data_ptr<double>(),
        l_discount.data_ptr<double>(),
        out.data_ptr<double>(),
        n
    );
    
    // Sort by revenue descending (small num_nations, so PyTorch sort is fine)
    auto values = out;
    auto indices = torch::arange(num_nations, out.options());
    auto sorted_result = values.sort(0, true);
    auto sorted_values = std::get<0>(sorted_result);
    auto sorted_indices = std::get<1>(sorted_result);
    auto sorted_nation_ids = indices.index_select(0, sorted_indices);
    
    return {sorted_nation_ids, sorted_values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_launcher", &q5_launcher, "TPC-H Q5 revenue aggregation (CUDA)");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q5_cuda_")
    cu_path = os.path.join(tmpdir, "_q5_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q5_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _launch_q5(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch CUDA kernel for Q5 revenue aggregation.
    
    GPU: Fused kernel computes revenue = price * (1 - discount) and atomically
    accumulates per nation_id. Sorting done in PyTorch (small num_nations).
    
    Why faster: Single CUDA pass replaces scatter_add_ + separate revenue compute.
    Coalesced memory access, atomicAdd per thread (acceptable for small num_nations ≤25).
    """
    result = _EXT.q5_launcher(nation_id, l_extendedprice, l_discount, num_nations)
    return result[0], result[1]

def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute revenue per nation and return sorted results.
    
    GPU: CUDA kernel fuses revenue computation and grouped aggregation.
    - Computes revenue = l_extendedprice * (1 - l_discount) per row
    - Atomically accumulates into output[nation_id] buffer
    - Returns (nation_ids, revenues) sorted by revenue descending
    
    Why faster: Single fused CUDA kernel vs. separate multiply + scatter_add.
    Memory-bound scan optimized via coalesced access and minimal HBM passes.
    """
    return _launch_q5(nation_id, l_extendedprice, l_discount, num_nations)


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
