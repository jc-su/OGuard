import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>

__global__ void q10_kernel(
    const int64_t* __restrict__ custkey_ptr,
    const double* __restrict__ extprice_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ rev_out,
    bool* __restrict__ present_out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = custkey_ptr[idx];
        double price = extprice_ptr[idx];
        double discount = discount_ptr[idx];
        double rev = price * (1.0 - discount);

        atomicAdd(&rev_out[key], rev);
        present_out[key] = true;
    }
}

void q10_launcher(
    torch::Tensor custkey,
    torch::Tensor extprice,
    torch::Tensor discount,
    torch::Tensor rev,
    torch::Tensor present
) {
    int64_t n = custkey.numel();
    if (n == 0) return;

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    q10_kernel<<<blocks, threads>>>(
        custkey.data_ptr<int64_t>(),
        extprice.data_ptr<double>(),
        discount.data_ptr<double>(),
        rev.data_ptr<double>(),
        present.data_ptr<bool>(),
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q10_launcher", &q10_launcher, "TPC-H Q10 core computation (CUDA)");
}
'''

def _load_cuda() -> object:
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
    Computes revenue per row and aggregates by c_custkey using a custom CUDA kernel.
    Because TPC-H keys are dense and bounded, we use a direct-addressing dense array
    strategy instead of a hash table for maximum memory bandwidth and speed.
    """
    if c_custkey.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=c_custkey.device),
            torch.empty(0, dtype=torch.float64, device=c_custkey.device)
        )
    
    max_key = int(c_custkey.max().item())
    
    # Failsafe for unexpected gigantic keys (> 100M keys = 800MB allocation)
    if max_key > 100_000_000:
        revenue_per_row = l_extendedprice * (1.0 - l_discount)
        unique_keys, inverse = torch.unique(c_custkey, return_inverse=True)
        revenue = torch.zeros(unique_keys.shape[0], dtype=torch.float64, device=revenue_per_row.device)
        revenue.scatter_add_(0, inverse, revenue_per_row)
        return unique_keys, revenue

    rev = torch.zeros(max_key + 1, dtype=torch.float64, device=c_custkey.device)
    present = torch.zeros(max_key + 1, dtype=torch.bool, device=c_custkey.device)
    
    # Ensure inputs are contiguous and on CUDA
    if not c_custkey.is_cuda:
        c_custkey = c_custkey.cuda()
    if not l_extendedprice.is_cuda:
        l_extendedprice = l_extendedprice.cuda()
    if not l_discount.is_cuda:
        l_discount = l_discount.cuda()
        
    c_custkey = c_custkey.contiguous()
    l_extendedprice = l_extendedprice.contiguous()
    l_discount = l_discount.contiguous()

    # Launch kernel to populate rev and present directly
    _EXT.q10_launcher(c_custkey, l_extendedprice, l_discount, rev, present)
    
    # Extract unique keys (which will naturally be sorted) and their accumulated revenues
    valid = present.nonzero(as_tuple=True)[0]
    final_rev = rev.index_select(0, valid)
    
    return valid, final_rev

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
