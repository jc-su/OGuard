import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void q10_revenue_kernel(
    const int64_t* __restrict__ custkey_ptr,
    const int64_t* __restrict__ inverse_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    double* __restrict__ revenue_ptr,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        double price = price_ptr[idx];
        double discount = discount_ptr[idx];
        double revenue = price * (1.0 - discount);
        int64_t out_idx = inverse_ptr[idx];
        atomicAdd(&revenue_ptr[out_idx], revenue);
    }
}

void q10_launcher(
    torch::Tensor custkey,
    torch::Tensor inverse,
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor revenue
) {
    TORCH_CHECK(custkey.scalar_type() == torch::kInt64, "custkey must be int64");
    TORCH_CHECK(inverse.scalar_type() == torch::kInt64, "inverse must be int64");
    TORCH_CHECK(price.scalar_type() == torch::kFloat64, "price must be float64");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(revenue.scalar_type() == torch::kFloat64, "revenue must be float64");
    
    if (!custkey.is_cuda()) custkey = custkey.cuda();
    if (!inverse.is_cuda()) inverse = inverse.cuda();
    if (!price.is_cuda()) price = price.cuda();
    if (!discount.is_cuda()) discount = discount.cuda();
    if (!revenue.is_cuda()) revenue = revenue.cuda();
    
    custkey = custkey.contiguous();
    inverse = inverse.contiguous();
    price = price.contiguous();
    discount = discount.contiguous();
    revenue = revenue.contiguous();
    
    int64_t n = custkey.numel();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    q10_revenue_kernel<<<blocks, BLOCK_SIZE>>>(
        custkey.data_ptr<int64_t>(),
        inverse.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        revenue.data_ptr<double>(),
        n
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q10_launcher", &q10_launcher, "TPC-H Q10 revenue aggregation (CUDA fused kernel)");
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
    Fused CUDA kernel for revenue calculation and group-by aggregation.
    
    GPU (CUDA): Computes revenue = price * (1 - discount) per row and aggregates
                by customer key using atomicAdd. This fuses projection + aggregation
                into one kernel pass, avoiding intermediate revenue_per_row tensor.
    
    Why faster: Memory-bound optimization - reduces HBM traffic by eliminating
                temporary tensor allocation and combining two operations into one
                kernel launch. Uses block-level parallelism with atomicAdd for
                thread-safe aggregation.
    
    Args:
        c_custkey: Customer keys (int64 tensor)
        l_extendedprice: Line item prices (float64 tensor)
        l_discount: Line item discounts (float64 tensor)
    
    Returns:
        Tuple of (unique_keys, aggregated_revenue) tensors
    """
    # Get unique keys and inverse mapping (torch unique is efficient)
    unique_keys, inverse = torch.unique(c_custkey, return_inverse=True)
    
    # Initialize revenue buffer for unique keys (zeros on GPU)
    revenue = torch.zeros(unique_keys.shape[0], dtype=torch.float64, device=c_custkey.device)
    
    # Launch fused CUDA kernel: compute revenue + scatter_add in one pass
    _EXT.q10_launcher(
        c_custkey,
        inverse,
        l_extendedprice,
        l_discount,
        revenue
    )
    
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
                 
                 Data flow:
                 - cuDF: Joins (customer↔orders↔lineitem↔nation), date/flag filters, string columns
                 - CUDA: Fused revenue calculation + group-by aggregation (memory-bound optimization)
                 - torch: Top-N selection on aggregated results
                 
                 Optimization: CUDA kernel fuses revenue projection (price*(1-discount)) with
                 scatter-add aggregation, eliminating intermediate tensor and reducing HBM traffic.
                 Uses atomicAdd for thread-safe parallel aggregation across 1024-thread blocks.
    
    Params:
        orderdate_lo (str): inclusive lower bound for o_orderdate.
        orderdate_hi (str): exclusive upper bound for o_orderdate.
        returnflag_val (str): lineitem return flag filter.
        top_n (int): number of top customers to return.
    
    Returns:
        cudf.DataFrame with columns: c_custkey, c_name, revenue, c_acctbal, n_name,
                                     c_address, c_phone, c_comment
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_nation = tables["nation"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter orders by date (cuDF)
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    # Filter lineitem by returnflag (cuDF)
    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with orders (cuDF)
    lo = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")

    # Join with customer (cuDF)
    loc = lo.merge(df_customer, left_on="o_custkey", right_on="c_custkey")

    # Aggregate revenue by c_custkey in CUDA (fused kernel)
    custkey_t = torch.as_tensor(loc["c_custkey"].to_cupy(), dtype=torch.int64)
    price_t = torch.as_tensor(loc["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(loc["l_discount"].to_cupy(), dtype=torch.float64)
    
    custkey_unique, revenue_t = _query_core(custkey_t, price_t, discount_t)

    # Top-N by revenue (torch)
    k = min(int(top_n), revenue_t.shape[0])
    top_rev, top_idx = torch.topk(revenue_t, k)
    top_keys = custkey_unique[top_idx]

    agg_df = cudf.DataFrame({
        "c_custkey": top_keys,
        "revenue": top_rev,
    })

    # Join back with customer to get descriptive cols (cuDF)
    cust_cols = df_customer[["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")

    # Join with nation (cuDF)
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]], left_on="c_nationkey", right_on="n_nationkey")

    # Order by revenue DESC (cuDF)
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
