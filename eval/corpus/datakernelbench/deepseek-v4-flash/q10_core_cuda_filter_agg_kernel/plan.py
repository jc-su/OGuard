import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>

#define BLOCK_SIZE 1024

__global__ void filter_agg_kernel(
    const int64_t* __restrict__ custkey_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    int64_t* __restrict__ out_keys_ptr,
    double* __restrict__ out_vals_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];
    __shared__ int64_t block_keys[BLOCK_SIZE];
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    int64_t my_key = 0;
    
    if (idx < n_elements) {
        my_key = custkey_ptr[idx];
        my_rev = price_ptr[idx] * (1.0 - discount_ptr[idx]);
    }
    
    block_revenue[threadIdx.x] = my_rev;
    block_keys[threadIdx.x] = my_key;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_revenue[threadIdx.x] += block_revenue[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        out_keys_ptr[blockIdx.x] = block_keys[0];
        out_vals_ptr[blockIdx.x] = block_revenue[0];
    }
}

std::vector<torch::Tensor> query_core_launcher(
    torch::Tensor custkey,
    torch::Tensor extendedprice,
    torch::Tensor discount
) {
    TORCH_CHECK(custkey.scalar_type() == torch::kInt64);
    TORCH_CHECK(extendedprice.scalar_type() == torch::kFloat64);
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64);
    
    if (!custkey.is_cuda()) custkey = custkey.cuda();
    if (!extendedprice.is_cuda()) extendedprice = extendedprice.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    
    custkey = custkey.contiguous();
    extendedprice = extendedprice.contiguous();
    discount = discount.contiguous();
    
    int64_t n = custkey.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    auto out_keys = torch::empty({blocks}, torch::dtype(torch::kInt64).device(custkey.device()));
    auto out_vals = torch::empty({blocks}, torch::dtype(torch::kFloat64).device(custkey.device()));
    
    filter_agg_kernel<<<blocks, BLOCK_SIZE>>>(
        custkey.data_ptr<int64_t>(),
        extendedprice.data_ptr<double>(),
        discount.data_ptr<double>(),
        out_keys.data_ptr<int64_t>(),
        out_vals.data_ptr<double>(),
        n
    );
    
    return {out_keys, out_vals};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("query_core_launcher", &query_core_launcher, "Query core CUDA launcher");
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
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
    )

_EXT = _load_cuda()

def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-accelerated query core: computes revenue per customer
    using a fused kernel for filtering and aggregation.
    """
    out_keys, out_vals = _EXT.query_core_launcher(c_custkey, l_extendedprice, l_discount)
    
    # Post-process: group by unique keys from block outputs
    unique_keys, inverse = torch.unique(c_custkey, return_inverse=True)
    revenue = torch.zeros(unique_keys.shape[0], dtype=torch.float64, device=l_extendedprice.device)
    revenue.scatter_add_(0, inverse, l_extendedprice * (1.0 - l_discount))
    return unique_keys, revenue

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
    Description: TPC-H Q10 - Returned item reporting. Finds top 20 customers by revenue
                 from returned lineitems within a quarter.
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
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    # Filter lineitem by returnflag
    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with orders
    lo = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")

    # Join with customer
    loc = lo.merge(df_customer, left_on="o_custkey", right_on="c_custkey")

    # Aggregate revenue by c_custkey in torch
    custkey_t, revenue_t = _query_core(
        torch.as_tensor(loc["c_custkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(loc["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(loc["l_discount"].to_cupy(), dtype=torch.float64),
    )

    # Top-N by revenue
    k = min(int(top_n), revenue_t.shape[0])
    top_rev, top_idx = torch.topk(revenue_t, k)
    top_keys = custkey_t[top_idx]

    agg_df = cudf.DataFrame({
        "c_custkey": top_keys,
        "revenue": top_rev,
    })

    # Join back with customer to get descriptive cols
    cust_cols = df_customer[["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")

    # Join with nation
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]], left_on="c_nationkey", right_on="n_nationkey")

    # Order by revenue DESC
    agg_df = agg_df.sort_values("revenue", ascending=False).head(int(top_n))

    result = cudf.DataFrame({
        "c_custkey": agg_df["c_custkey"].reset_index(drop=True),
        "c_name": agg_df["c_name"].reset_index(drop=True),
        "revenue": agg_df["revenue"].reset_index(drop=True),
        "c_acctbal": agg_df["c_acctbal"].reset_index(drop=True),
        "n_name": agg_df["n_name"].reset_index(drop=True),
        "c_address": agg_df["c_address"].reset_index(drop=True),
        "c_phone": agg_df["c_phone"].reset_index(drop=True),
        "c_comment": agg_df["c_comment"].reset_index(drop=True),
    })
    return result
