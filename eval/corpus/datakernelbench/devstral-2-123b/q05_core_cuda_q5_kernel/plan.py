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
    const double* __restrict__ extendedprice_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ nation_id_ptr,
    double* __restrict__ revenue_ptr,
    int64_t num_nations,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double rev = extendedprice_ptr[idx] * (1.0 - discount_ptr[idx]);
        int64_t nation_id = nation_id_ptr[idx];
        if (nation_id >= 0 && nation_id < num_nations) {
            atomicAdd(&revenue_ptr[nation_id], rev);
        }
    }
}

extern "C" void q5_launcher(
    torch::Tensor extendedprice,
    torch::Tensor discount,
    torch::Tensor nation_id,
    torch::Tensor out_revenue,
    int64_t num_nations
) {
    TORCH_CHECK(extendedprice.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    TORCH_CHECK(nation_id.scalar_type() == torch::kLong);
    TORCH_CHECK(out_revenue.scalar_type() == torch::kFloat64);
    if (!extendedprice.is_cuda()) extendedprice = extendedprice.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!nation_id.is_cuda()) nation_id = nation_id.cuda();
    if (!out_revenue.is_cuda()) out_revenue = out_revenue.cuda();
    extendedprice = extendedprice.contiguous();
    discount = discount.contiguous();
    nation_id = nation_id.contiguous();
    out_revenue = out_revenue.contiguous();
    int64_t n = extendedprice.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMemset(out_revenue.data_ptr<double>(), 0, num_nations * sizeof(double));
    q5_kernel<<<blocks, BLOCK_SIZE>>>(
        extendedprice.data_ptr<double>(),
        discount.data_ptr<double>(),
        nation_id.data_ptr<int64_t>(),
        out_revenue.data_ptr<double>(),
        num_nations,
        n
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q5_launcher", &q5_launcher, "TPC-H Q5 revenue (CUDA)");
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

def _query_core(
    nation_id: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    num_nations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = l_extendedprice.device
    dtype = torch.float64
    out_rev = torch.zeros(num_nations, dtype=dtype, device=device)
    _EXT.q5_launcher(
        l_extendedprice,
        l_discount,
        nation_id,
        out_rev,
        num_nations,
    )
    nation_ids = torch.arange(num_nations, dtype=torch.int64, device=device)
    order = torch.argsort(out_rev, descending=True)
    return nation_ids[order], out_rev[order]

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
