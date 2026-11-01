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

@triton.jit
def _filter_kernel(
    l_shipdate_ptr, o_orderdate_ptr,
    shipdate_lo, orderdate_hi,
    out_valid_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=0)
    orderdate = tl.load(o_orderdate_ptr + offs, mask=mask, other=0)
    
    valid = (shipdate > shipdate_lo) & (orderdate < orderdate_hi)
    tl.store(out_valid_ptr + offs, valid.to(tl.int8), mask=mask)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q3_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _query_core(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = l_orderkey.numel()
    device = l_orderkey.device
    
    # Triton kernel for filtering (fused date predicate check)
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Allocate validity mask
    valid_mask = torch.empty(n, dtype=torch.int8, device=device)
    
    _K._filter_kernel[grid](
        l_shipdate_days, o_orderdate_days,
        shipdate_lo_days, orderdate_hi_days,
        valid_mask, n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Convert to boolean and get indices
    valid_bool = valid_mask.to(torch.bool)
    valid_indices = torch.nonzero(valid_bool, as_tuple=False).squeeze(1)
    
    if valid_indices.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )
    
    # Filter using indices
    lk = l_orderkey[valid_indices]
    od = o_orderdate_days[valid_indices]
    sp = o_shippriority[valid_indices]
    rev = l_extendedprice[valid_indices] * (1.0 - l_discount[valid_indices])
    
    # Group By: (l_orderkey, o_orderdate, o_shippriority)
    keys = torch.stack([lk.to(torch.int64), od.to(torch.int64), sp.to(torch.int64)], dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    
    # Aggregate revenue
    revenue_sum = torch.zeros(unique_keys.shape[0], dtype=rev.dtype, device=device)
    revenue_sum.scatter_add_(0, inverse, rev)
    
    u_lk = unique_keys[:, 0]
    u_od = unique_keys[:, 1].to(torch.int32)
    u_sp = unique_keys[:, 2].to(torch.int32)
    
    # Sort: revenue DESC, o_orderdate ASC
    n_unique = unique_keys.shape[0]
    k = min(top_n, n_unique)
    
    # Stable sort by secondary key first (o_orderdate ASC)
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]
    
    # Stable sort by primary key (revenue DESC)
    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    
    lk_sorted = lk_s[idx_pri][:k]
    rev_sorted = rev_s[idx_pri][:k]
    od_sorted = od_s[idx_pri][:k]
    sp_sorted = sp_s[idx_pri][:k]
    
    return lk_sorted, rev_sorted, od_sorted, sp_sorted


def run_query(
    tables: dict[str, cudf.DataFrame],
    mktsegment: str = "BUILDING",
    orderdate_hi: str = "1995-03-15",
    shipdate_lo: str = "1995-03-15",
    top_n: int = 10,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate,
               o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < CAST('1995-03-15' AS date)
          AND l_shipdate > CAST('1995-03-15' AS date)
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;
    Description: TPC-H Q3 - Shipping Priority. Joins customer, orders, lineitem;
                 filters by market segment and date predicates; aggregates revenue
                 per (orderkey, orderdate, shippriority); returns top-N by revenue.
    Params:
        mktsegment (str): customer market segment filter.
        orderdate_hi (str): exclusive upper bound on o_orderdate.
        shipdate_lo (str): exclusive lower bound on l_shipdate.
        top_n (int): LIMIT value.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Literal normalization: string dates -> epoch-day ints
    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    # Filter customers by mktsegment
    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]

    # Join customer -> orders
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")

    # Join (customer, orders) -> lineitem
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Convert date columns to int32 day-counts
    o_orderdate_days = (
        (col["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    l_shipdate_days = (
        (col["l_shipdate"].astype("int64") // 86400).astype("int32")
    )

    lk_t, rev_t, od_t, sp_t = _query_core(
        torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32),
        shipdate_lo_days,
        orderdate_hi_days,
        int(top_n),
    )

    # Convert orderdate back to datetime64[s] for output
    od_series = cudf.Series(od_t.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_t,
        "revenue": rev_t,
        "o_orderdate": od_series,
        "o_shippriority": sp_t,
    })
