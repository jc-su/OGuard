import os
import tempfile

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source – all device code and PyBind11 bindings are embedded here.
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// ----------------------------------------------------------------------
// One-pass kernel: compute (price * (1 - discount)) per row and
// atomically add it into a per‑customer accumulator.
// ----------------------------------------------------------------------
__global__ void group_rev_kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ out_rev,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t key = custkey[idx];
    double rev = price[idx] * (1.0 - discount[idx]);

    // atomicAdd on double is supported on sm_70+ (H100 is sm_90)
    atomicAdd(&out_rev[key], rev);
}

// ----------------------------------------------------------------------
// Launcher: allocates a dense revenue buffer sized to (max custkey + 1),
// runs the kernel, and returns the buffer as a torch Tensor.
// ----------------------------------------------------------------------
torch::Tensor group_rev_launcher(
    torch::Tensor custkey,
    torch::Tensor price,
    torch::Tensor discount)
{
    TORCH_CHECK(custkey.is_cuda() && price.is_cuda() && discount.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(custkey.dtype() == torch::kInt64,
                "custkey must be int64");
    TORCH_CHECK(price.dtype() == torch::kFloat64 && discount.dtype() == torch::kFloat64,
                "price and discount must be float64");

    // Ensure contiguous tensors for simple pointer arithmetic.
    custkey = custkey.contiguous();
    price   = price.contiguous();
    discount = discount.contiguous();

    int64_t n = custkey.numel();

    // Determine the largest customer key to size the output buffer.
    int64_t max_key = custkey.max().item<int64_t>();

    // Allocate a dense buffer of size (max_key + 1), zero‑initialised.
    auto out = torch::zeros({max_key + 1}, price.options());

    int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    group_rev_kernel<<<blocks, BLOCK_SIZE>>>(
        custkey.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        out.data_ptr<double>(),
        n);

    // Synchronise to surface any launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in group_rev_kernel launch: %s\n",
               cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_rev_launcher", &group_rev_launcher,
          "Fused revenue projection + group‑by for TPC‑H Q10 (CUDA)");
}
'''

def _load_cuda():
    """Compile the embedded CUDA source and load the resulting extension."""
    tmpdir = tempfile.mkdtemp(prefix="q10_cuda_")
    cu_path = os.path.join(tmpdir, "q10_kernels.cu")
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
        verbose=False,
    )

# Load the compiled CUDA extension once at import time.
_EXT = _load_cuda()


def _launch_group_rev(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> torch.Tensor:
    """
    Invoke the fused CUDA kernel that computes per‑row revenue
    ``price * (1 - discount)`` and aggregates it by ``c_custkey``.
    Returns a dense tensor ``out_rev`` where ``out_rev[customer_id]``
    holds the summed revenue (zero for customers without rows).
    """
    return _EXT.group_rev_launcher(c_custkey, l_extendedprice, l_discount)


def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute revenue = l_extendedprice * (1 - l_discount) and aggregate
    by c_custkey using a single fused CUDA kernel.

    The kernel performs:
      * Coalesced loads of the three columns.
      * Per‑row revenue calculation.
      * Atomic accumulation into a dense per‑customer buffer.

    After the kernel we extract the non‑zero entries to obtain the compact
    ``unique_keys`` and matching ``revenue`` tensors, matching the reference
    implementation's output layout.
    """
    # Run fused kernel → dense per‑customer revenue buffer.
    dense_rev = _launch_group_rev(c_custkey, l_extendedprice, l_discount)

    # Extract indices of customers with non‑zero revenue.
    nonzero_idx = torch.nonzero(dense_rev, as_tuple=False).squeeze(1)
    revenue = dense_rev[nonzero_idx]
    return nonzero_idx, revenue


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

    # Aggregate revenue by c_custkey in CUDA
    custkey_t, revenue_t = _query_core(
        torch.as_tensor(loc["c_custkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(loc["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(loc["l_discount"].to_cupy(), dtype=torch.float64),
    )

    # Top‑N by revenue
    k = min(int(top_n), revenue_t.shape[0])
    top_rev, top_idx = torch.topk(revenue_t, k)
    top_keys = custkey_t[top_idx]

    agg_df = cudf.DataFrame({
        "c_custkey": top_keys,
        "revenue": top_rev,
    })

    # Join back with customer to retrieve descriptive columns
    cust_cols = df_customer[[
        "c_custkey", "c_name", "c_acctbal", "c_phone",
        "c_address", "c_comment", "c_nationkey"
    ]]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")

    # Join with nation for nation name
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]],
                         left_on="c_nationkey", right_on="n_nationkey")

    # Order by revenue DESC and keep only the requested number of rows
    agg_df = agg_df.sort_values("revenue", ascending=False).head(int(top_n))

    result = cudf.DataFrame({
        "c_custkey": agg_df["c_custkey"].reset_index(drop=True),
        "c_name":    agg_df["c_name"].reset_index(drop=True),
        "revenue":   agg_df["revenue"].reset_index(drop=True),
        "c_acctbal": agg_df["c_acctbal"].reset_index(drop=True),
        "n_name":    agg_df["n_name"].reset_index(drop=True),
        "c_address": agg_df["c_address"].reset_index(drop=True),
        "c_phone":   agg_df["c_phone"].reset_index(drop=True),
        "c_comment": agg_df["c_comment"].reset_index(drop=True),
    })
    return result
