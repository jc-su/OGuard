import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
import numpy as np

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _q3_kernel(
    l_orderkey_ptr, l_extendedprice_ptr, l_discount_ptr,
    l_shipdate_days_ptr, o_orderdate_days_ptr, o_shippriority_ptr,
    shipdate_lo_days, orderdate_hi_days,
    out_rev_ptr, out_lk_ptr, out_od_ptr, out_sp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    l_ship = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_lo_days)
    o_order = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=orderdate_hi_days)

    keep = mask & (l_ship > shipdate_lo_days) & (o_order < orderdate_hi_days)

    ep = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    rev = tl.where(keep, ep * (1.0 - disc), 0.0)
    lk = tl.load(l_orderkey_ptr + offs, mask=mask, other=0)
    od = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=0)
    sp = tl.load(o_shippriority_ptr + offs, mask=mask, other=0)
    
    tl.store(out_rev_ptr + offs, rev, mask=mask)
    tl.store(out_lk_ptr + offs, lk, mask=mask)
    tl.store(out_od_ptr + offs, od, mask=mask)
    tl.store(out_sp_ptr + offs, sp, mask=mask)
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
    Description: TPC-H Q3 - Shipping Priority.
                 cuDF handles joins and string filtering. A fused Triton kernel
                 applies date predicates and projects revenue in a single pass.
                 PyTorch handles filtering zeros, grouping, and sorting.
    Params:
        mktsegment (str): customer market segment filter.
        orderdate_hi (str): exclusive upper bound on o_orderdate.
        shipdate_lo (str): exclusive lower bound on l_shipdate.
        top_n (int): LIMIT value.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    # cuDF: filter customer by mktsegment, then join orders and lineitem
    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # Convert date columns to int32 day-counts
    o_orderdate_days = (col["o_orderdate"].astype("int64") // 86400).astype("int32")
    l_shipdate_days = (col["l_shipdate"].astype("int64") // 86400).astype("int32")

    # Convert to GPU tensors
    l_orderkey = torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_extendedprice = torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount = torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    o_orderdate = torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32)
    o_shippriority = torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32)

    device = l_orderkey.device
    n = l_orderkey.numel()

    # Fused kernel: filter + project (keeps all rows, non-kept have zero revenue)
    out_rev = torch.empty(n, dtype=torch.float64, device=device)
    out_lk = torch.empty(n, dtype=torch.int64, device=device)
    out_od = torch.empty(n, dtype=torch.int32, device=device)
    out_sp = torch.empty(n, dtype=torch.int32, device=device)

    BLOCK_SIZE = 16384
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _K._q3_kernel[grid](
        l_orderkey, l_extendedprice, l_discount,
        l_shipdate, o_orderdate, o_shippriority,
        shipdate_lo_days, orderdate_hi_days,
        out_rev, out_lk, out_od, out_sp, n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Filter out rows with zero revenue (didn't pass predicates)
    mask = out_rev > 0.0
    if not mask.any():
        return cudf.DataFrame({
            "l_orderkey": cudf.Series([], dtype="int64"),
            "revenue": cudf.Series([], dtype="float64"),
            "o_orderdate": cudf.Series([], dtype="datetime64[s]"),
            "o_shippriority": cudf.Series([], dtype="int32"),
        })

    rev_filtered = out_rev[mask]
    lk_filtered = out_lk[mask]
    od_filtered = out_od[mask]
    sp_filtered = out_sp[mask]

    # Group by unique composite key using PyTorch
    keys = torch.stack([
        lk_filtered.to(torch.int64), 
        od_filtered.to(torch.int64), 
        sp_filtered.to(torch.int64)
    ], dim=1)
    
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

    revenue_sum = torch.zeros(unique_keys.shape[0], dtype=rev_filtered.dtype, device=device)
    revenue_sum.scatter_add_(0, inverse, rev_filtered)

    u_lk = unique_keys[:, 0]
    u_od = unique_keys[:, 1].to(torch.int32)
    u_sp = unique_keys[:, 2].to(torch.int32)

    # Sort: revenue DESC, o_orderdate ASC, limit top_n
    n_unq = unique_keys.shape[0]
    k = min(top_n, n_unq)

    # Stable sort by o_orderdate ASC first, then revenue DESC
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]

    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    rev_out = rev_s[idx_pri][:k]
    lk_out = lk_s[idx_pri][:k]
    od_out = od_s[idx_pri][:k]
    sp_out = sp_s[idx_pri][:k]

    # Convert orderdate back to datetime64[s]
    od_cpu = od_out.cpu().numpy().astype(np.int64) * 86400
    od_series = cudf.Series(od_cpu).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_out.cpu().numpy(),
        "revenue": rev_out.cpu().numpy(),
        "o_orderdate": od_series,
        "o_shippriority": sp_out.cpu().numpy(),
    })
