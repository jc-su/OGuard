import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _filter_proj_kernel(
    l_orderkey_ptr, l_extendedprice_ptr, l_discount_ptr,
    l_shipdate_ptr, o_orderdate_ptr, o_shippriority_ptr,
    rev_out_ptr,
    shipdate_lo, orderdate_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    l_ship = tl.load(l_shipdate_ptr + offs, mask=mask, other=orderdate_hi)
    o_date = tl.load(o_orderdate_ptr + offs, mask=mask, other=shipdate_lo)

    keep = mask & (l_ship > shipdate_lo) & (o_date < orderdate_hi)

    lk = tl.load(l_orderkey_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.int64))
    od = tl.load(o_orderdate_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.int32))
    sp = tl.load(o_shippriority_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.int32))
    ep = tl.load(l_extendedprice_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.float64))
    disc = tl.load(l_discount_ptr + offs, mask=keep, other=tl.zeros([BLOCK_SIZE], dtype=tl.float64))

    rev = tl.where(keep, ep * (1.0 - disc), tl.zeros([BLOCK_SIZE], dtype=tl.float64))

    # Store filtered outputs back (masked rows become 0)
    tl.store(l_orderkey_ptr + offs, tl.where(keep, lk, tl.zeros([BLOCK_SIZE], dtype=tl.int64)), mask=mask)
    tl.store(o_orderdate_ptr + offs, tl.where(keep, od, tl.zeros([BLOCK_SIZE], dtype=tl.int32)), mask=mask)
    tl.store(o_shippriority_ptr + offs, tl.where(keep, sp, tl.zeros([BLOCK_SIZE], dtype=tl.int32)), mask=mask)
    tl.store(rev_out_ptr + offs, rev, mask=mask)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q3_triton_")
    path = os.path.join(tmpdir, "_q3_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q3_kernels", path)
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
    """
    GPU-accelerated Q3 core: Triton kernel does fused filter+project into output buffers.
    Then PyTorch handles group-by aggregation and top-N sort on GPU.
    """
    n = l_orderkey.shape[0]
    device = l_orderkey.device

    # Clone inputs to avoid modifying originals
    lk = l_orderkey.clone()
    od = o_orderdate_days.clone()
    sp = o_shippriority.clone()

    # Allocate output revenue buffer (zeroed)
    rev_out = torch.zeros(n, dtype=torch.float64, device=device)

    # Launch fused kernel
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._filter_proj_kernel[grid](
        lk, l_extendedprice, l_discount,
        l_shipdate_days, od, sp,
        rev_out,
        shipdate_lo_days, orderdate_hi_days,
        n,
        BLOCK_SIZE=4096,
    )

    # Filter out rows where orderkey == 0 (filtered rows)
    valid_mask = lk != 0
    lk_f = lk[valid_mask]
    od_f = od[valid_mask].to(torch.int64)
    sp_f = sp[valid_mask].to(torch.int64)
    rev_f = rev_out[valid_mask]

    if lk_f.numel() == 0:
        # No matching rows
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        return empty, empty, empty.to(torch.int32), empty.to(torch.int32)

    # Group by composite key: (l_orderkey, o_orderdate, o_shippriority)
    keys = torch.stack([lk_f, od_f, sp_f], dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    revenue_sum = torch.zeros(unique_keys.shape[0], dtype=torch.float64, device=device)
    revenue_sum.scatter_add_(0, inverse, rev_f)

    u_lk = unique_keys[:, 0]
    u_od = unique_keys[:, 1].to(torch.int32)
    u_sp = unique_keys[:, 2].to(torch.int32)

    # ORDER BY revenue DESC, o_orderdate ASC LIMIT top_n
    n_unique = unique_keys.shape[0]
    k = min(top_n, n_unique)

    # Stable sort by secondary key first (o_orderdate ASC)
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]

    # Then sort by primary key (revenue DESC)
    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    rev_sorted = rev_s[idx_pri][:k]
    lk_sorted = lk_s[idx_pri][:k]
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
