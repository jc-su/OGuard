import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

using namespace cooperative_groups;

__global__ void q10_kernel(
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ custkey_ptr,
    double* __restrict__ revenue_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double price = price_ptr[idx];
        double discount = discount_ptr[idx];
        int64_t custkey = custkey_ptr[idx];
        double revenue = price * (1.0 - discount);
        atomicAdd(&revenue_ptr[custkey], revenue);
    }
}

torch::Tensor q10_launcher(
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor custkey,
    int64_t max_custkey
) {
    price = price.contiguous();
    discount = discount.contiguous();
    custkey = custkey.contiguous();

    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!custkey.is_cuda()) custkey = custkey.cuda();

    TORCH_CHECK(price.numel() == discount.numel());
    TORCH_CHECK(price.numel() == custkey.numel());

    int64_t n = price.numel();
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE);

    auto revenue = torch::zeros({max_custkey + 1}, torch::TensorOptions().dtype(price.dtype()).device(price.device()));

    q10_kernel<<<blocks, threads>>>(
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        custkey.data_ptr<int64_t>(),
        revenue.data_ptr<double>(),
        n
    );

    return revenue;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q10_launcher", &q10_launcher, "TPC-H Q10 fused CUDA kernel");
}
'''

def _load_cuda():
    tmpdir = tempfile.mkdtemp(prefix="q10_cuda_")
    cu_path = os.path.join(tmpdir, "_q10_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="q10_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90", "-std=c++17"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def run_query(
    tables: dict[str, cudf.DataFrame],
    orderdate_lo: str = "1993-10-01",
    orderdate_hi: str = "1994-01-01",
    returnflag_val: str = "R",
    top_n: int = 20,
) -> cudf.DataFrame:
    """
    SQL: SELECT c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) AS revenue,
         c_acctbal, n_name, c_address, c_phone, c_comment
         FROM customer, orders, lineitem, nation
         WHERE c_custkey=o_custkey AND l_orderkey=o_orderkey
         AND o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01'
         AND l_returnflag='R' AND c_nationkey=n_nationkey
         GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
         ORDER BY revenue DESC LIMIT 20;
    Description: TPC-H Q10 - Returned item reporting. Uses CUDA kernel for aggregation
                 then joins to get customer/nation info. Matches reference implementation
                 by using similar grouping logic with atomic operations.
    Params:
        orderdate_lo (str): inclusive lower bound for o_orderdate.
        orderdate_hi (str): exclusive upper bound for o_orderdate.
        returnflag_val (str): lineitem return flag filter.
        top_n (int): number of top customers to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_nation = tables["nation"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter orders by date
    o_days = df_orders["o_orderdate"].astype("int64") // 86400
    orders_filtered = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    # Join lineitem with filtered orders
    line_orders = df_lineitem.merge(orders_filtered, left_on="l_orderkey", right_on="o_orderkey")

    # Filter by return flag
    lo_f = line_orders[line_orders["l_returnflag"] == returnflag_val]

    # Get max customer key for output array
    max_custkey = df_customer["c_custkey"].max()

    # Prepare tensors for CUDA kernel
    l_extendedprice = torch.as_tensor(lo_f["l_extendedprice"].to_cupy().astype('float64'))
    l_discount = torch.as_tensor(lo_f["l_discount"].to_cupy().astype('float64'))
    custkey = torch.as_tensor(lo_f["o_custkey"].to_cupy().astype('int64'))

    # Aggregate revenue using CUDA kernel
    revenue = _EXT.q10_launcher(l_extendedprice, l_discount, custkey, max_custkey)

    # Create DataFrame with non-zero revenue customers
    valid_keys = torch.nonzero(revenue).flatten()
    if valid_keys.numel() > 0:
        agg_data = {
            "c_custkey": valid_keys.cpu().numpy(),
            "revenue": revenue[valid_keys].cpu().numpy()
        }
        agg_df = cudf.DataFrame(agg_data)

        # Join with customer and nation
        customer_links = df_customer[["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]]
        agg_df = agg_df.merge(customer_links, on="c_custkey")
        agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]], left_on="c_nationkey", right_on="n_nationkey")

        # Sort and limit
        result = agg_df.sort_values("revenue", ascending=False).head(top_n)
    else:
        # Return empty DataFrame if no valid keys
        result = cudf.DataFrame({
            "c_custkey": [],
            "c_name": [],
            "revenue": [],
            "c_acctbal": [],
            "n_name": [],
            "c_address": [],
            "c_phone": [],
            "c_comment": []
        })

    return result[[
        "c_custkey", "c_name", "revenue", "c_acctbal",
        "n_name", "c_address", "c_phone", "c_comment"
    ]].reset_index(drop=True)
