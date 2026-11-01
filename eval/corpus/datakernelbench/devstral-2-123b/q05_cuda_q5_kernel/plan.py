import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

__global__ void q5_kernel(
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    const int64_t* __restrict__ nation_id_ptr,
    double* __restrict__ out_revenue,
    int64_t num_nations,
    int64_t num_rows
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rows) {
        int64_t nation_id = nation_id_ptr[idx];
        if (nation_id >= 0 && nation_id < num_nations) {
            double revenue = l_extendedprice[idx] * (1.0 - l_discount[idx]);
            atomicAdd(&out_revenue[nation_id], revenue);
        }
    }
}

void q5_launcher(
    torch::Tensor l_extendedprice_t,
    torch::Tensor l_discount_t,
    torch::Tensor nation_id_t,
    torch::Tensor out_revenue_t,
    int64_t num_nations,
    int64_t num_rows
) {
    // Validate input types and device
    TORCH_CHECK(l_extendedprice_t.scalar_type() == torch::kFloat64);
    TORCH_CHECK(l_discount_t.scalar_type() == torch::kFloat64);
    TORCH_CHECK(nation_id_t.scalar_type() == torch::kInt64);
    TORCH_CHECK(out_revenue_t.scalar_type() == torch::kFloat64);

    // Ensure tensors are on CUDA and contiguous
    l_extendedprice_t = l_extendedprice_t.contiguous().cuda();
    l_discount_t = l_discount_t.contiguous().cuda();
    nation_id_t = nation_id_t.contiguous().cuda();
    out_revenue_t = out_revenue_t.contiguous().cuda();

    // Zero out the output
    auto out_rev_view = out_revenue_t.view({-1});
    out_rev_view.zero_();

    // Calculate grid size and launch kernel
    int64_t blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    q5_kernel<<<blocks, BLOCK_SIZE>>>(
        l_extendedprice_t.data_ptr<double>(),
        l_discount_t.data_ptr<double>(),
        nation_id_t.data_ptr<int64_t>(),
        out_revenue_t.data_ptr<double>(),
        num_nations,
        num_rows
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_launcher", &q5_launcher, "TPC-H Q5 fused aggregation");
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
    Description: TPC-H Q5 with optimized CUDA kernel for group-by aggregation.
        Each thread directly computes the revenue and atomically adds it to the
        corresponding nation bucket. This simplifies the accumulation logic.
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

    # Filter and join operations
    region_f = df_region[df_region["r_name"] == region_name]
    nation_f = df_nation.merge(region_f[["r_regionkey"]], left_on="n_regionkey", right_on="r_regionkey")

    orders_f = df_orders.copy()
    orders_f["o_orderdate_days"] = (orders_f["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = orders_f[(orders_f["o_orderdate_days"] >= lo_days) & (orders_f["o_orderdate_days"] < hi_days)]

    co = df_customer[["c_custkey", "c_nationkey"]].merge(
        orders_f[["o_orderkey", "o_custkey"]], left_on="c_custkey", right_on="o_custkey"
    )

    col = co.merge(
        df_lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]],
        left_on="o_orderkey", right_on="l_orderkey"
    )

    cols = col.merge(
        df_supplier[["s_suppkey", "s_nationkey"]],
        left_on="l_suppkey", right_on="s_suppkey"
    )
    cols = cols[cols["c_nationkey"] == cols["s_nationkey"]]

    final_df = cols.merge(
        nation_f[["n_nationkey", "n_name"]],
        left_on="s_nationkey", right_on="n_nationkey"
    )

    # Create nation id mapping
    nations_unique = nation_f[["n_nationkey", "n_name"]].drop_duplicates().sort_values("n_nationkey").reset_index(drop=True)
    nations_unique["nation_id"] = cudf.Series(range(len(nations_unique)), dtype="int64")
    final_df = final_df.merge(
        nations_unique[["n_nationkey", "nation_id"]],
        on="n_nationkey"
    )

    num_nations = int(len(nations_unique))

    # Prepare GPU tensors
    l_extendedprice_t = torch.as_tensor(final_df["l_extendedprice"].to_cupy(), dtype=torch.float64).cuda()
    l_discount_t = torch.as_tensor(final_df["l_discount"].to_cupy(), dtype=torch.float64).cuda()
    nation_id_t = torch.as_tensor(final_df["nation_id"].to_cupy(), dtype=torch.int64).cuda()
    num_rows = int(len(final_df))

    # Initialize output tensor
    out_revenue_t = torch.zeros(num_nations, dtype=torch.float64, device="cuda")

    # Launch CUDA kernel
    _EXT.q5_launcher(
        l_extendedprice_t,
        l_discount_t,
        nation_id_t,
        out_revenue_t,
        num_nations,
        num_rows
    )

    # Get results and map to nation names
    result = cudf.DataFrame({
        "n_name": nations_unique["n_name"].to_numpy(),
        "revenue": out_revenue_t.cpu().numpy()
    })

    # Sort by revenue descending
    result = result.sort_values("revenue", ascending=False).reset_index(drop=True)

    return result
