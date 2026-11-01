import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <algorithm>

#define BLOCK_SIZE 1024

__global__ void q5_kernel(
    const int64_t* __restrict__ nation_id,
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    double* __restrict__ out,
    int64_t n_elements,
    int num_nations,
    int num_copies
) {
    extern __shared__ double block_revenue[]; 

    int tid = threadIdx.x;
    int total_bins = num_copies * num_nations;
    // Initialize shared memory
    for (int i = tid; i < total_bins; i += blockDim.x) {
        block_revenue[i] = 0.0;
    }
    __syncthreads();

    // Process elements and accumulate in thread-specific shared memory bank copies
    // to reduce atomicAdd contention.
    int64_t idx = blockIdx.x * blockDim.x + tid;
    if (idx < n_elements) {
        int64_t nid = nation_id[idx];
        if (nid >= 0 && nid < num_nations) {
            double rev = extendedprice[idx] * (1.0 - discount[idx]);
            int copy_idx = tid % num_copies;
            atomicAdd(&block_revenue[copy_idx * num_nations + nid], rev);
        }
    }
    __syncthreads();

    // Reduce copies and write to global memory
    for (int nid = tid; nid < num_nations; nid += blockDim.x) {
        double sum = 0.0;
        for (int c = 0; c < num_copies; ++c) {
            sum += block_revenue[c * num_nations + nid];
        }
        if (sum != 0.0) {
            atomicAdd(&out[nid], sum);
        }
    }
}

torch::Tensor q5_launcher(
    torch::Tensor nation_id,
    torch::Tensor l_extendedprice,
    torch::Tensor l_discount,
    int num_nations
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

    int64_t n = l_extendedprice.numel();
    auto out = torch::zeros({num_nations}, l_extendedprice.options());

    if (n > 0 && num_nations > 0) {
        int max_shared_mem = 49152; // 48 KB
        int max_copies = 8;
        int max_possible_copies = max_shared_mem / (num_nations * sizeof(double));
        int num_copies = std::min(max_copies, max_possible_copies);
        if (num_copies < 1) num_copies = 1;

        size_t shared_mem_size = num_copies * num_nations * sizeof(double);
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        q5_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>(
            nation_id.data_ptr<int64_t>(),
            l_extendedprice.data_ptr<double>(),
            l_discount.data_ptr<double>(),
            out.data_ptr<double>(),
            n,
            num_nations,
            num_copies
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_launcher", &q5_launcher, "TPC-H Q5 revenue (CUDA)");
}
'''

def _load_cuda():
    """
    Writes and JIT-compiles the CUDA source for the hot path.
    Targeting Hopper (sm_90) as per GPU spec.
    """
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

def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core GPU execution using fused CUDA kernel.
    Replaces eager scatter_add with highly optimized shared memory reduction.
    """
    out = _EXT.q5_launcher(
        nation_id,
        l_extendedprice,
        l_discount,
        num_nations
    )
    nation_ids = torch.arange(num_nations, dtype=torch.int64, device=out.device)
    # Sort by revenue desc
    order = torch.argsort(out, descending=True)
    return nation_ids[order], out[order]

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
