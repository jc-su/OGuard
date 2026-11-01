import os
import tempfile
import types
import torch
from torch.utils.cpp_extension import load
import cudf
import pandas as pd

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024
#define MAX_NATIONS 25

__global__ void q5_revenue_kernel(
    const int64_t* __restrict__ nation_ids,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ out,
    int64_t n_rows,
    int num_nations
) {
    __shared__ double block_agg[MAX_NATIONS];

    // Zero the shared accumulator for this block
    if (threadIdx.x < num_nations) {
        block_agg[threadIdx.x] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_rows) {
        int nid = static_cast<int>(nation_ids[idx]);
        double rev = price[idx] * (1.0 - discount[idx]);
        atomicAdd(&block_agg[nid], rev);
    }
    __syncthreads();

    // Each thread that corresponds to a nation writes its block sum to global output
    if (threadIdx.x < num_nations) {
        atomicAdd(&out[threadIdx.x], block_agg[threadIdx.x]);
    }
}

torch::Tensor q5_launcher(
    torch::Tensor nation_ids,
    torch::Tensor price,
    torch::Tensor discount,
    int num_nations
) {
    TORCH_CHECK(nation_ids.scalar_type() == torch::kInt64, "nation_ids must be int64");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");

    if (!nation_ids.is_cuda()) nation_ids = nation_ids.cuda();
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();

    nation_ids = nation_ids.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();

    int64_t n = nation_ids.numel();
    auto out = torch::zeros({num_nations}, price.options()); // float64

    if (n == 0) {
        // Nothing to aggregate; return zeros
        return out;
    }

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Shared memory size: num_nations * sizeof(double)
    int shared_mem = num_nations * sizeof(double);
    q5_revenue_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>(
        nation_ids.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        out.data_ptr<double>(),
        n,
        num_nations
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_launcher", &q5_launcher, "TPC-H Q5 revenue aggregation (CUDA)");
}
'''

def _load_cuda() -> types.ModuleType:
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

# Load the compiled extension once at import time
_EXT = _load_cuda()


def _query_core_cuda(
    nation_ids: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> torch.Tensor:
    """
    Compute revenue per nation using a fused CUDA kernel.
    Returns a 1‑D tensor of shape (num_nations,) with float64 revenue.
    """
    return _EXT.q5_launcher(nation_ids, l_extendedprice, l_discount, num_nations)


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

    Description: TPC‑H Q5 – local supplier volume for a region over one year.
    The hot path (filter, project, aggregate) is offloaded to a hand‑written CUDA
    kernel that computes revenue = price * (1 - discount) and aggregates by nation
    in a single fused pass.  All joins and string handling stay on cuDF.

    Params:
        region_name (str): region to filter on (default 'ASIA').
        orderdate_lo (str): inclusive lower bound on order date (default '1994-01-01').
        orderdate_hi (str): exclusive upper bound on order date (default '1995-01-01').
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    # Convert date bounds to integer days since epoch (int32)
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter region
    region_f = df_region[df_region["r_name"] == region_name]

    # Nations belonging to the filtered region
    nation_f = df_nation.merge(
        region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey"
    )

    # Filter orders by date range and keep needed columns
    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (
        orders_f["o_orderdate"].astype("int64") // 86400
    ).astype("int32")
    orders_f = orders_f[
        (orders_f["o_orderdate_days"] >= lo_days)
        & (orders_f["o_orderdate_days"] < hi_days)
    ]

    # Join customer with filtered orders
    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f[["o_orderkey", "o_custkey"]], left_on="c_custkey", right_on="o_custkey"
    )

    # Join with lineitem
    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey",
        right_on="l_orderkey",
    )

    # Join with supplier, keeping only matches where customer nation equals supplier nation
    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey",
        right_on="s_suppkey",
    )
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    # Join with filtered nation to obtain nation name
    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey",
        right_on="n_nationkey",
    )

    # If no rows after joins, return empty result matching the expected schema
    if final_df.empty:
        return cudf.DataFrame({"n_name": [], "revenue": []})

    # Build mapping from n_nationkey to a compact nation_id (0..num_nations-1)
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

    num_nations = len(nations_unique)

    # Move columns to GPU tensors (float64 for price/discount, int64 for nation ids)
    nation_ids_t = torch.as_tensor(
        final_df["nation_id"].to_cupy(), dtype=torch.int64
    ).cuda()
    price_t = torch.as_tensor(
        final_df["l_extendedprice"].to_cupy(), dtype=torch.float64
    ).cuda()
    discount_t = torch.as_tensor(
        final_df["l_discount"].to_cupy(), dtype=torch.float64
    ).cuda()

    # Run the fused CUDA kernel to compute revenue per nation
    revenue_t = _query_core_cuda(
        nation_ids_t, price_t, discount_t, num_nations
    )

    # Pull result back to host for sorting and building the final DataFrame
    revenue_np = revenue_t.cpu().numpy()

    # Create a list of (revenue, name) using the same ordering as nation_id
    # Use to_arrow().to_pylist() instead of tolist() which is not supported by cuDF
    n_name_by_id = nations_unique["n_name"].to_arrow().to_pylist()

    result_data = {
        "n_name": [n_name_by_id[i] for i in range(num_nations)],
        "revenue": [revenue_np[i] for i in range(num_nations)],
    }
    result_df = cudf.DataFrame(result_data)
    # Sort by revenue descending
    result_df = result_df.sort_values("revenue", ascending=False).reset_index(
        drop=True
    )

    return result_df
