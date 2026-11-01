import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <stdint.h>

#define BLOCK_SIZE 1024

__global__ void agg_revenue_kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ out,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t key = keys[idx];
        double rev = price[idx] * (1.0 - discount[idx]);
        atomicAdd(&out[key], rev);
    }
}

torch::Tensor agg_revenue_launcher(
    torch::Tensor keys,
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor out
) {
    TORCH_CHECK(keys.is_cuda(), "keys must be a CUDA tensor");
    TORCH_CHECK(price.is_cuda(), "price must be a CUDA tensor");
    TORCH_CHECK(discount.is_cuda(), "discount must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");

    keys = keys.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();
    out = out.contiguous();

    int64_t n = keys.numel();
    int64_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    agg_revenue_kernel<<<blocks, BLOCK_SIZE>>>(
        keys.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        out.data_ptr<double>(),
        n
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("agg_revenue", &agg_revenue_launcher, "Aggregate revenue by customer key (CUDA)");
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
    Aggregates revenue (l_extendedprice * (1 - l_discount)) per customer key using a fused CUDA kernel.
    Returns (unique_keys, revenue) for keys with non‑zero revenue.
    """
    if c_custkey.numel() == 0:
        # Empty input: return empty tensors
        return (
            torch.tensor([], dtype=torch.int64, device=c_custkey.device),
            torch.tensor([], dtype=torch.float64, device=c_custkey.device),
        )

    # Ensure inputs are on CUDA - is_cuda is a property, not a method
    if not c_custkey.is_cuda:
        c_custkey = c_custkey.cuda()
    if not l_extendedprice.is_cuda:
        l_extendedprice = l_extendedprice.cuda()
    if not l_discount.is_cuda:
        l_discount = l_discount.cuda()

    # Allocate output array indexed by customer key
    max_key = int(c_custkey.max().item())
    out = torch.zeros(max_key + 1, dtype=torch.float64, device=c_custkey.device)

    # Launch fused kernel: compute revenue and aggregate via atomicAdd
    _EXT.agg_revenue(c_custkey, l_extendedprice, l_discount, out)

    # Extract only keys with non‑zero revenue
    mask = out != 0
    idx = torch.nonzero(mask, as_tuple=True)[0]
    revenue = out[idx]
    return idx, revenue


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
                 from returned lineitems within a quarter. This implementation uses a fused
                 CUDA kernel for the hot‑path aggregation (filter → project → sum) to reduce
                 memory traffic and eliminate intermediate tensors.

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

    # Convert dates to int32 epoch days (int32) for fast comparison
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter orders by order date
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    # Filter lineitem by return flag
    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with orders
    lo = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")

    # Join with customer
    loc = lo.merge(df_customer, left_on="o_custkey", right_on="c_custkey")

    # Prepare GPU tensors for the hot aggregation
    c_custkey_t = torch.as_tensor(loc["c_custkey"].to_cupy(), dtype=torch.int64)
    l_extendedprice_t = torch.as_tensor(loc["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(loc["l_discount"].to_cupy(), dtype=torch.float64)

    # Aggregate revenue per customer using the fused CUDA kernel
    custkey_unique, revenue_unique = _query_core(c_custkey_t, l_extendedprice_t, l_discount_t)

    # Top N customers by revenue
    k = min(int(top_n), revenue_unique.shape[0])
    top_rev, top_idx = torch.topk(revenue_unique, k)
    top_keys = custkey_unique[top_idx]

    # Build a small DataFrame with the aggregated results
    agg_df = cudf.DataFrame({
        "c_custkey": top_keys.cpu().numpy(),
        "revenue": top_rev.cpu().numpy(),
    })

    # Join back with customer to retrieve descriptive columns
    cust_cols = df_customer[["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")

    # Join with nation to get nation name
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]], left_on="c_nationkey", right_on="n_nationkey")

    # Order by revenue descending and keep only top N
    agg_df = agg_df.sort_values("revenue", ascending=False).head(int(top_n))

    # Produce final result with exact column order
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
