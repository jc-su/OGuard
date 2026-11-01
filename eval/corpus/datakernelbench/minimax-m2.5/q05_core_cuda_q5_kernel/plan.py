import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024
#define MAX_NATIONS 32

__global__ void q5_kernel(
    const int64_t* __restrict__ nation_ids,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    int num_nations,
    double* __restrict__ out,
    int64_t n
) {
    // Shared memory for per‑nation block‑level partial sums
    __shared__ double block_sums[MAX_NATIONS];

    // Zero the shared memory for the relevant nations
    if (threadIdx.x < MAX_NATIONS) {
        block_sums[threadIdx.x] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int nid = static_cast<int>(nation_ids[idx]); // 0 .. num_nations-1
        double rev = price[idx] * (1.0 - discount[idx]);
        if (nid >= 0 && nid < num_nations) {
            atomicAdd(&block_sums[nid], rev);
        }
    }
    __syncthreads();

    // Write block partial sums to global output (one atomic per nation)
    if (threadIdx.x < num_nations) {
        atomicAdd(&out[threadIdx.x], block_sums[threadIdx.x]);
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

    // Output tensor of shape (num_nations,) filled with zeros
    auto out = torch::zeros({num_nations}, price.options().dtype(torch::kFloat64));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int shared_mem = MAX_NATIONS * sizeof(double);

    q5_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>(
        nation_ids.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        num_nations,
        out.data_ptr<double>(),
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_launcher", &q5_launcher, "TPC‑H Q5 CUDA kernel");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q5_cuda_")
    cu_path = os.path.join(tmpdir, "q5_kernels.cu")
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


def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU‑accelerated aggregation for Q5.
    Input:
        nation_id   – int64 tensor (size = number of filtered rows)
        l_extendedprice – float64 tensor (same size)
        l_discount      – float64 tensor (same size)
        num_nations     – number of distinct nations in the filter
    Returns:
        sorted_nation_ids (int64) and sorted_revenue (float64), ordered by
        revenue descending. This matches the reference `_query_core` signature.
    """
    # Ensure data lives on the GPU and is contiguous
    if not nation_id.is_cuda:
        nation_id = nation_id.cuda()
    if not l_extendedprice.is_cuda:
        l_extendedprice = l_extendedprice.cuda()
    if not l_discount.is_cuda:
        l_discount = l_discount.cuda()

    nation_id = nation_id.contiguous()
    l_extendedprice = l_extendedprice.contiguous()
    l_discount = l_discount.contiguous()

    # Launch the fused CUDA kernel
    revenue_out = _EXT.q5_launcher(nation_id, l_extendedprice, l_discount, num_nations)

    # Sort by revenue descending (GPU‐based argsort)
    order = torch.argsort(revenue_out, descending=True)
    sorted_nation_ids = torch.arange(num_nations, dtype=torch.int64, device=revenue_out.device)[order]
    sorted_revenue = revenue_out[order]

    return sorted_nation_ids, sorted_revenue


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
