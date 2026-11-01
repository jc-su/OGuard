import os
import tempfile
import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------
# CUDA source: fused aggregation kernel (group‑by revenue per customer)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// ------------------------------------------------------------------
// Kernel: each thread processes one row, atomically adds its revenue
// to the per‑customer bucket (customer keys are dense 0..max_key)
// ------------------------------------------------------------------
__global__ void agg_kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    double* __restrict__ revenue_out,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t key = custkey[idx];
    double rev = price[idx] * (1.0 - discount[idx]);
    // atomicAdd on double is supported on sm_70+ (H100 = sm_90)
    atomicAdd(&revenue_out[key], rev);
}

// ------------------------------------------------------------------
// Launcher: checks, makes everything contiguous and launches kernel
// ------------------------------------------------------------------
void agg_launcher(
    torch::Tensor custkey,
    torch::Tensor price,
    torch::Tensor discount,
    torch::Tensor revenue_out,
    int64_t n)
{
    TORCH_CHECK(custkey.is_cuda(), "custkey must be CUDA");
    TORCH_CHECK(price.is_cuda(),   "price must be CUDA");
    TORCH_CHECK(discount.is_cuda(),"discount must be CUDA");
    TORCH_CHECK(revenue_out.is_cuda(), "revenue_out must be CUDA");

    TORCH_CHECK(custkey.dtype() == torch::kInt64,   "custkey dtype must be int64");
    TORCH_CHECK(price.dtype()   == torch::kFloat64, "price dtype must be float64");
    TORCH_CHECK(discount.dtype()== torch::kFloat64, "discount dtype must be float64");
    TORCH_CHECK(revenue_out.dtype()==torch::kFloat64,"revenue_out dtype must be float64");

    custkey   = custkey.contiguous();
    price     = price.contiguous();
    discount  = discount.contiguous();
    revenue_out = revenue_out.contiguous();

    const int threads = BLOCK_SIZE;
    const int blocks  = (n + threads - 1) / threads;

    agg_kernel<<<blocks, threads>>>(
        custkey.data_ptr<int64_t>(),
        price.data_ptr<double>(),
        discount.data_ptr<double>(),
        revenue_out.data_ptr<double>(),
        n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("agg_launcher", &agg_launcher, "Aggregates revenue per customer (CUDA)");
}
'''

def _load_cuda():
    """Compile the CUDA source on first import and cache the built module."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q10_cuda_")
    cu_path = os.path.join(tmpdir, "q10_kernels.cu")
    build_dir = os.path.join(tmpdir, "build")
    os.makedirs(build_dir, exist_ok=True)
    with open(cu_path, "w") as f:
        f.write(_CUDA_SOURCE)
    return load(
        name="tpch_q10_cuda_ext",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )

# Load (or compile) the extension exactly once per interpreter session
_EXT = _load_cuda()

# ----------------------------------------------------------------------
# GPU‑accelerated aggregation core (level core)
# ----------------------------------------------------------------------
def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform ``revenue = sum(l_extendedprice * (1 - l_discount))`` grouped by
    ``c_custkey`` using a fused CUDA kernel.

    The kernel atomically adds each row's contribution to a dense buffer
    indexed by the customer key (keys are 1‑based and dense up to the max key
    present).  After the kernel finishes, the non‑zero entries are extracted
    to produce the ``unique_keys`` and ``revenue`` tensors required by the
    rest of the query pipeline.

    This replaces the two‑step ``torch.unique`` + ``scatter_add_`` approach
    with a single pass, reducing memory traffic and kernel launch overhead.
    """
    # Determine the size of the dense revenue buffer (max key + 1, 0‑based)
    max_key = int(c_custkey.max().item())
    revenue_buf = torch.zeros(max_key + 1, dtype=torch.float64, device=c_custkey.device)

    # Launch fused aggregation kernel
    _EXT.agg_launcher(
        c_custkey,
        l_extendedprice,
        l_discount,
        revenue_buf,
        c_custkey.shape[0],
    )

    # Extract only customers that actually contributed revenue
    nonzero_mask = revenue_buf != 0.0
    unique_keys = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(1)
    revenue = revenue_buf[unique_keys]

    return unique_keys, revenue

# ----------------------------------------------------------------------
# Public query entry point (unchanged signature)
# ----------------------------------------------------------------------
def run_query(
    tables: dict[str, cudf.DataFrame],
    orderdate_lo: str = "1993-10-01",
    orderdate_hi: str = "1994-01-01",
    returnflag_val: str = "R",
    top_n: int = 20,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) AS revenue,
               c_acctbal, n_name, c_address, c_phone, c_comment
        FROM   customer, orders, lineitem, nation
        WHERE  c_custkey = o_custkey
          AND  l_orderkey = o_orderkey
          AND  o_orderdate >= '1993-10-01'
          AND  o_orderdate <  '1994-01-01'
          AND  l_returnflag = 'R'
          AND  c_nationkey = n_nationkey
        GROUP BY c_custkey, c_name, c_acctbal, c_phone,
                 n_name, c_address, c_comment
        ORDER BY revenue DESC
        LIMIT 20;

    Description:
        The query filters orders by a date range and lineitems by return flag,
        joins the four tables, and computes revenue per customer.
        The expensive part – revenue aggregation – is performed by a
        fused CUDA kernel that atomically accumulates revenue into a dense
        per‑customer buffer, eliminating multiple passes and temporary
        tensors that the reference implementation incurs.

    Params:
        orderdate_lo (str): inclusive lower bound for o_orderdate.
        orderdate_hi (str): exclusive upper bound for o_orderdate.
        returnflag_val (str): lineitem return flag filter.
        top_n (int): number of top customers to return.
    """
    # ------------------------------------------------------------------
    # 1️⃣  cuDF preprocessing: filters + joins
    # ------------------------------------------------------------------
    df_customer = tables["customer"]
    df_orders   = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_nation   = tables["nation"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter orders by date (convert to epoch‑days int32)
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    # Filter lineitem by return flag
    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem → orders → customer
    lo = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")
    loc = lo.merge(df_customer, left_on="o_custkey", right_on="c_custkey")

    # ------------------------------------------------------------------
    # 2️⃣  Transfer to GPU tensors (cupy → torch) – zero copy via DLPack
    # ------------------------------------------------------------------
    custkey_t = torch.as_tensor(
        loc["c_custkey"].to_cupy(), dtype=torch.int64
    )
    price_t = torch.as_tensor(
        loc["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    discount_t = torch.as_tensor(
        loc["l_discount"].to_cupy(), dtype=torch.float64
    )

    # ------------------------------------------------------------------
    # 3️⃣  GPU aggregation (fused kernel)
    # ------------------------------------------------------------------
    cust_keys_agg, revenue_agg = _query_core(custkey_t, price_t, discount_t)

    # ------------------------------------------------------------------
    # 4️⃣  Top‑N by revenue (still on GPU, using torch)
    # ------------------------------------------------------------------
    k = min(int(top_n), revenue_agg.shape[0])
    top_rev, top_idx = torch.topk(revenue_agg, k)
    top_keys = cust_keys_agg[top_idx]

    agg_df = cudf.DataFrame({"c_custkey": top_keys, "revenue": top_rev})

    # ------------------------------------------------------------------
    # 5️⃣  Attach remaining descriptive columns (cuDF joins)
    # ------------------------------------------------------------------
    cust_cols = df_customer[
        ["c_custkey", "c_name", "c_acctbal", "c_phone",
         "c_address", "c_comment", "c_nationkey"]
    ]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]],
                          left_on="c_nationkey", right_on="n_nationkey")

    # ------------------------------------------------------------------
    # 6️⃣  Final ordering & column layout
    # ------------------------------------------------------------------
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
