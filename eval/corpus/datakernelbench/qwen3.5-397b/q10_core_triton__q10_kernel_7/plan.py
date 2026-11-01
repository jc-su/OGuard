import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import triton.language as tl
import cudf
import pandas as pd

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
    reset_to_zero=["revenue_ptr"],
)
@triton.jit
def _q10_kernel(
    extendedprice_ptr,
    discount_ptr,
    inverse_ptr,
    revenue_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: compute revenue = extendedprice * (1 - discount) and scatter-add to output.
    Memory-bound optimization: single pass through input, no intermediate tensor allocation.
    Uses atomic_add for group-by aggregation via inverse indices from torch.unique.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load inputs with bounds checking
    extendedprice = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    inverse = tl.load(inverse_ptr + offs, mask=mask, other=0)
    
    # Compute revenue per row: extendedprice * (1 - discount)
    revenue = extendedprice * (1.0 - discount)
    
    # Scatter-add to output revenue array using inverse indices for group-by
    tl.atomic_add(revenue_ptr + inverse, revenue, mask=mask)
'''

def _load_kernels():
    """
    Load Triton kernels from a temporary on-disk file.
    Required because @triton.jit needs real file paths for compilation.
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

def _launch_q10(
    extendedprice: torch.Tensor,
    discount: torch.Tensor,
    inverse: torch.Tensor,
    num_unique: int,
) -> torch.Tensor:
    """
    Launch the fused Q10 aggregation kernel.
    
    Args:
        extendedprice: 1D tensor of line item extended prices (float64)
        discount: 1D tensor of line item discounts (float64)
        inverse: 1D tensor of inverse indices from torch.unique (int64)
        num_unique: Number of unique customer keys (size of output revenue array)
    
    Returns:
        revenue: 1D tensor of aggregated revenue per unique customer (float64)
    """
    n = extendedprice.numel()
    revenue = torch.zeros(num_unique, dtype=torch.float64, device=extendedprice.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q10_kernel[grid](
        extendedprice,
        discount,
        inverse,
        revenue,
        n,
    )
    return revenue

def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute revenue aggregation grouped by customer key.
    
    GPU execution: Triton fused kernel computes revenue = extendedprice * (1 - discount)
    and scatter-adds to output array using inverse indices from torch.unique.
    
    Why faster than reference:
    - Fused compute: revenue calculation + scatter-add in single kernel pass
    - No intermediate tensor: eliminates revenue_per_row temporary allocation
    - Memory-bound optimization: reduces HBM traffic by ~50% (no write/read of intermediate)
    - Autotuned block sizes for H100 bandwidth optimization
    
    Args:
        c_custkey: 1D tensor of customer keys (int64)
        l_extendedprice: 1D tensor of line item extended prices (float64)
        l_discount: 1D tensor of line item discounts (float64)
    
    Returns:
        unique_keys: 1D tensor of unique customer keys (int64)
        revenue: 1D tensor of aggregated revenue per unique customer (float64)
    """
    # Get unique customer keys and inverse mapping for group-by
    unique_keys, inverse = torch.unique(c_custkey, return_inverse=True)
    
    # Launch fused Triton kernel for revenue computation + aggregation
    revenue = _launch_q10(
        l_extendedprice,
        l_discount,
        inverse,
        unique_keys.shape[0],
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
