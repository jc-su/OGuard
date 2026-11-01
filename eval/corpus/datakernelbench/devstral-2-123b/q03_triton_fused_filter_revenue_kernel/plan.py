import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
from typing import Tuple

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def fused_filter_revenue_kernel(
    l_orderkey_ptr, l_extendedprice_ptr, l_discount_ptr, l_shipdate_ptr,
    o_orderdate_ptr, o_shippriority_ptr,
    shipdate_lo, orderdate_hi,
    n_elements,
    out_lk_ptr, out_od_ptr, out_sp_ptr, out_rev_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all columns
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=shipdate_lo)
    orderdate = tl.load(o_orderdate_ptr + offs, mask=mask, other=orderdate_hi)
    lk = tl.load(l_orderkey_ptr + offs, mask=mask, other=-1)
    price = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    sp = tl.load(o_shippriority_ptr + offs, mask=mask, other=-1)

    # Apply predicates
    keep = mask & (shipdate > shipdate_lo) & (orderdate < orderdate_hi)

    # Store kept values for later aggregation
    base = tl.program_id(0) * BLOCK_SIZE
    tl.store(out_lk_ptr + base + tl.arange(0, BLOCK_SIZE), lk, mask=keep)
    tl.store(out_od_ptr + base + tl.arange(0, BLOCK_SIZE), orderdate, mask=keep)
    tl.store(out_sp_ptr + base + tl.arange(0, BLOCK_SIZE), sp, mask=keep)
    tl.store(out_rev_ptr + base + tl.arange(0, BLOCK_SIZE),
             tl.where(keep, price * (1.0 - discount), 0.0), mask=keep)
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

def _launch_fused_filter_agg(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    orderdate_hi_days: int,
    top_n: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = l_orderkey.numel()
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    # Pre-allocate output tensors
    out_lk = torch.full([n], -1, dtype=torch.int64, device=l_orderkey.device)
    out_od = torch.full([n], 9999999, dtype=torch.int64, device=l_orderkey.device)
    out_sp = torch.full([n], -1, dtype=torch.int64, device=l_orderkey.device)
    out_rev = torch.full([n], 0.0, dtype=torch.float64, device=l_orderkey.device)

    _K.fused_filter_revenue_kernel[grid](
        l_orderkey, l_extendedprice, l_discount, l_shipdate_days,
        o_orderdate_days, o_shippriority,
        shipdate_lo_days, orderdate_hi_days,
        n,
        out_lk, out_od, out_sp, out_rev,
        BLOCK_SIZE,
    )

    # Filter valid rows
    valid = (out_lk != -1) & (out_od != 9999999) & (out_sp != -1)
    valid_idx = torch.nonzero(valid).squeeze()

    if len(valid_idx) == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=l_orderkey.device),
            torch.empty(0, dtype=torch.float64, device=l_orderkey.device),
            torch.empty(0, dtype=torch.int64, device=l_orderkey.device),
            torch.empty(0, dtype=torch.int32, device=l_orderkey.device)
        )

    # Create dataframe for grouping
    df = cudf.DataFrame({
        'lk': out_lk[valid_idx].cpu().numpy(),
        'od': out_od[valid_idx].cpu().numpy(),
        'sp': out_sp[valid_idx].cpu().numpy(),
        'rev': out_rev[valid_idx].cpu().numpy()
    })

    # Group by keys and sum revenue
    grouped = df.groupby(['lk', 'od', 'sp'])['rev'].sum().reset_index()

    # Sort and select top-N
    grouped['sort1'] = -grouped['rev']  # DESC
    grouped['sort2'] = grouped['od']      # ASC
    sorted_df = grouped.sort_values(['sort1', 'sort2']).head(top_n)

    # Convert to tensors
    lk_sorted = torch.as_tensor(sorted_df['lk'].values, dtype=torch.int64, device=l_orderkey.device)
    rev_sorted = torch.as_tensor(sorted_df['rev'].values, dtype=torch.float64, device=l_orderkey.device)
    od_sorted = torch.as_tensor(sorted_df['od'].values, dtype=torch.int64, device=l_orderkey.device)
    sp_sorted = torch.as_tensor(sorted_df['sp'].values, dtype=torch.int32, device=l_orderkey.device)

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
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    cust_f = df_customer[df_customer["c_mktsegment"] == mktsegment][["c_custkey"]]
    co = cust_f.merge(df_orders, left_on="c_custkey", right_on="o_custkey")
    col = co.merge(df_lineitem, left_on="o_orderkey", right_on="l_orderkey")

    o_orderdate_days = ((col["o_orderdate"].astype("int64") // 86400).astype("int32"))
    l_shipdate_days = ((col["l_shipdate"].astype("int64") // 86400).astype("int32"))

    lk_t, rev_t, od_t, sp_t = _launch_fused_filter_agg(
        torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32),
        torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32),
        int(shipdate_lo_days),
        int(orderdate_hi_days),
        int(top_n),
    )

    od_series = cudf.Series(od_t.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_t,
        "revenue": rev_t,
        "o_orderdate": od_series,
        "o_shippriority": sp_t,
    })
