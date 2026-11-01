import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["agg_revenue_ptr"],
)
@triton.jit
def _q10_aggregate_kernel(
    extendedprice_ptr,
    discount_ptr,
    inverse_ptr,
    agg_revenue_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q10 revenue aggregation.
    
    Computes revenue = extendedprice * (1 - discount) per row,
    then atomically accumulates into per-customer buckets using inverse indices.
    
    Memory-bound optimization: single pass through data, revenue calc + agg fused,
    avoiding intermediate tensor materialization from reference torch.scatter_add_.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load extendedprice and discount (fp64 for precision)
    ext_price = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    
    # Compute revenue in registers: l_extendedprice * (1 - l_discount)
    revenue = ext_price * (1.0 - discount)
    
    # Load inverse indices (mapping each row to unique customer bucket 0..num_unique-1)
    inverse = tl.load(inverse_ptr + offs, mask=mask, other=0)
    
    # Atomic add to per-customer aggregation buffer
    # Each thread block processes BLOCK_SIZE rows, atomically accumulating
    tl.atomic_add(agg_revenue_ptr + inverse, revenue, mask=mask)
'''


def _load_kernels():
    """
    Load Triton kernels from disk.
    
    Triton requires @triton.jit functions to be in actual .py files.
    We write _KERNEL_SOURCE to a temp file and import it dynamically.
    """
    tmpdir = tempfile.mkdtemp(prefix="q10_triton_")
    path = os.path.join(tmpdir, "_q10_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q10_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_aggregate(
    extendedprice: torch.Tensor,
    discount: torch.Tensor,
    inverse: torch.Tensor,
    num_unique: int,
) -> torch.Tensor:
    """
    Launch Triton kernel for fused revenue aggregation.
    
    Args:
        extendedprice: l_extendedprice values (float64)
        discount: l_discount values (float64)
        inverse: inverse indices from torch.unique (int64)
        num_unique: number of unique customer keys
    
    Returns:
        agg_revenue: aggregated revenue per unique customer (float64)
    """
    n = extendedprice.numel()
    agg_revenue = torch.zeros(num_unique, dtype=torch.float64, device=extendedprice.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q10_aggregate_kernel[grid](
        extendedprice,
        discount,
        inverse,
        agg_revenue,
        n,
    )
    return agg_revenue


def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute revenue aggregation by customer using fused Triton kernel.
    
    GPU execution: Triton kernel fuses revenue calculation and group-by aggregation
    in a single pass, reducing HBM traffic vs reference torch.unique + scatter_add_.
    
    Args:
        c_custkey: customer keys (int64)
        l_extendedprice: lineitem extended prices (float64)
        l_discount: lineitem discounts (float64)
    
    Returns:
        unique_keys: unique customer keys
        revenue: aggregated revenue per customer
    """
    # Get unique customer keys and inverse mapping (needed for aggregation buckets)
    unique_keys, inverse = torch.unique(c_custkey, return_inverse=True)
    
    # Fused Triton kernel: revenue calc + atomic aggregation in one pass
    revenue = _launch_aggregate(
        l_extendedprice,
        l_discount,
        inverse,
        num_unique=unique_keys.shape[0],
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
                 
                 GPU acceleration: cuDF handles joins and string columns (memory-efficient
                 for variable-length data). Triton fused kernel handles the hot path:
                 revenue calculation + group-by aggregation in single pass, reducing
                 HBM traffic vs reference torch.unique + scatter_add_ approach.
                 
                 Memory-bound optimization: fusion avoids materializing intermediate
                 revenue_per_row tensor, computing revenue in registers before atomic add.
    
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

    # Convert date bounds to int32 epoch days for efficient comparison
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter orders by date range (cuDF optimized for datetime operations)
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    # Filter lineitem by returnflag (string comparison in cuDF)
    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with orders (l_orderkey = o_orderkey)
    lo = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")

    # Join with customer (o_custkey = c_custkey)
    loc = lo.merge(df_customer, left_on="o_custkey", right_on="c_custkey")

    # Extract tensors for GPU aggregation (Triton hot path)
    # Use float64 for revenue calculation to match reference precision
    custkey_t = torch.as_tensor(loc["c_custkey"].to_cupy(), dtype=torch.int64)
    extprice_t = torch.as_tensor(loc["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(loc["l_discount"].to_cupy(), dtype=torch.float64)

    # Fused Triton aggregation: revenue calc + group-by sum
    custkey_unique, revenue_t = _query_core(custkey_t, extprice_t, discount_t)

    # Top-N by revenue (torch.topk is efficient on GPU)
    k = min(int(top_n), revenue_t.shape[0])
    top_rev, top_idx = torch.topk(revenue_t, k)
    top_keys = custkey_unique[top_idx]

    # Build aggregation result dataframe
    agg_df = cudf.DataFrame({
        "c_custkey": top_keys,
        "revenue": top_rev,
    })

    # Join back with customer to get descriptive columns
    cust_cols = df_customer[["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")

    # Join with nation to get nation name
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]], left_on="c_nationkey", right_on="n_nationkey")

    # Sort by revenue DESC and take top N (already top-k but ensure order)
    agg_df = agg_df.sort_values("revenue", ascending=False).head(int(top_n))

    # Build final result with exact column order from SELECT
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
