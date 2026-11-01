import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import triton.language as tl
import cudf
import pandas as pd
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# Triton Kernel Source
# All @triton.jit code must reside in this string to satisfy the validator.
# BLOCK_SIZE is fixed (not autotuned) to ensure consistency across two-pass filter.
# -----------------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

BLOCK_SIZE = 4096

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["block_counts_ptr"],
)
@triton.jit
def _count_valid_kernel(
    sd_ptr, od_ptr,
    block_counts_ptr,
    sd_lo, od_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    sd = tl.load(sd_ptr + offs, mask=mask, other=sd_lo - 1)
    od = tl.load(od_ptr + offs, mask=mask, other=od_hi + 1)
    keep = (sd > sd_lo) & (od < od_hi) & mask
    count = tl.sum(keep.to(tl.int32))
    tl.store(block_counts_ptr + pid, count)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
)
@triton.jit
def _filter_pack_kernel(
    lk_ptr, ep_ptr, disc_ptr, sd_ptr, od_ptr, sp_ptr,
    out_keys_ptr, out_rev_ptr,
    block_offsets_ptr,
    sd_lo, od_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    sd = tl.load(sd_ptr + offs, mask=mask, other=sd_lo - 1)
    od = tl.load(od_ptr + offs, mask=mask, other=od_hi + 1)
    keep = (sd > sd_lo) & (od < od_hi) & mask
    
    # Load data needed for valid rows
    lk = tl.load(lk_ptr + offs, mask=mask, other=0)
    ep = tl.load(ep_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    sp = tl.load(sp_ptr + offs, mask=mask, other=0)
    
    # Calculate revenue
    rev = ep * (1.0 - disc)
    
    # Pack Key: (lk << 32) | (od << 16) | sp
    # Cast to int64 for shifting
    lk_i64 = lk.to(tl.int64)
    od_i64 = od.to(tl.int64)
    sp_i64 = sp.to(tl.int64)
    
    packed_key = (lk_i64 << 32) | (od_i64 << 16) | sp_i64
    
    # Compaction using cumsum
    local_idx = tl.cumsum(keep.to(tl.int32)) - 1
    
    # Global index
    base_offset = tl.load(block_offsets_ptr + pid)
    global_idx = base_offset + local_idx
    
    # Store only if keep
    tl.store(out_keys_ptr + global_idx, packed_key, mask=keep)
    tl.store(out_rev_ptr + global_idx, rev, mask=keep)
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

def _run_triton_filter(
    lk: torch.Tensor,
    ep: torch.Tensor,
    disc: torch.Tensor,
    sd: torch.Tensor,
    od: torch.Tensor,
    sp: torch.Tensor,
    sd_lo: int,
    od_hi: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Runs the fused filter and pack kernel (two-pass for correct compaction).
    Returns: packed_keys, revenues, total_valid_count
    """
    n = lk.numel()
    device = lk.device
    
    if n == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            0
        )
    
    # Fixed BLOCK_SIZE for both passes to ensure consistency
    BLOCK_SIZE = 4096
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    grid = (grid_size,)
    
    # Pass 1: Count valid per block
    block_counts = torch.zeros(grid_size, dtype=torch.int32, device=device)
    
    _K._count_valid_kernel[grid](
        sd, od, block_counts,
        sd_lo, od_hi,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Host-side prefix sum to get offsets
    # offsets[i] = sum(counts[:i+1]) (end index for block i)
    offsets = torch.cumsum(block_counts, dim=0, dtype=torch.int32)
    total_count = int(offsets[-1].item())
    
    if total_count == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            0
        )
    
    # Allocate output buffers
    out_keys = torch.empty(total_count, dtype=torch.int64, device=device)
    out_rev = torch.empty(total_count, dtype=torch.float64, device=device)
    
    # Calculate start_offsets for each block
    # start_offsets[i] = offsets[i-1] for i > 0, else 0
    start_offsets = torch.zeros_like(offsets)
    if grid_size > 1:
        start_offsets[1:] = offsets[:-1]
    
    # Pass 2: Write compacted data
    _K._filter_pack_kernel[grid](
        lk, ep, disc, sd, od, sp,
        out_keys, out_rev,
        start_offsets,
        sd_lo, od_hi,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_keys, out_rev, total_count

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
    Optimized core using Triton for filter/pack and PyTorch for agg/sort.
    """
    device = l_orderkey.device
    
    # 1. Fused Filter + Revenue + Key Pack (Triton)
    packed_keys, revenues, count = _run_triton_filter(
        l_orderkey, l_extendedprice, l_discount,
        l_shipdate_days, o_orderdate_days, o_shippriority,
        shipdate_lo_days, orderdate_hi_days
    )
    
    if count == 0:
        # Return empty tensors with correct dtypes
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )
    
    # 2. Aggregation (Group By Sum)
    # Use torch.unique on packed keys (1D int64 is faster than 2D stack)
    unique_keys, inverse = torch.unique(packed_keys, return_inverse=True)
    
    revenue_sum = torch.zeros(unique_keys.shape[0], dtype=torch.float64, device=device)
    revenue_sum.scatter_add_(0, inverse, revenues)
    
    # 3. Unpack Keys
    # packed = (lk << 32) | (od << 16) | sp
    u_lk = (unique_keys >> 32).to(torch.int64)
    u_od = ((unique_keys >> 16) & 0xFFFF).to(torch.int32)
    u_sp = (unique_keys & 0xFFFF).to(torch.int32)
    
    # 4. Sort: Revenue DESC, OrderDate ASC
    # Use two-pass stable sort (matching reference approach)
    # Sort by secondary key first (o_orderdate ASC), then primary (revenue DESC)
    n = unique_keys.shape[0]
    k = min(top_n, n)
    
    # Sort by secondary first (o_orderdate ASC) - stable
    idx_sec = torch.argsort(u_od, stable=True)
    rev_s = revenue_sum[idx_sec]
    lk_s = u_lk[idx_sec]
    od_s = u_od[idx_sec]
    sp_s = u_sp[idx_sec]
    
    # Sort by primary (revenue DESC) - stable
    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    
    # Select Top-K
    rev_sorted = rev_s[idx_pri][:k]
    lk_sorted = lk_s[idx_pri][:k]
    od_sorted = od_s[idx_pri][:k]
    sp_sorted = sp_s[idx_pri][:k]
    
    return (
        lk_sorted,
        rev_sorted,
        od_sorted,
        sp_sorted,
    )

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
    Description: TPC-H Q3 optimized with Triton. 
                 Joins performed in cuDF. 
                 Hot path (filter, project, key-pack) fused in Triton to reduce HBM passes.
                 Aggregation uses torch.unique (1D packed keys) + scatter_add.
                 Sorting uses two-pass stable argsort (secondary then primary).
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

    # Prepare tensors
    lk_t = torch.as_tensor(col["l_orderkey"].to_cupy(), dtype=torch.int64)
    ep_t = torch.as_tensor(col["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(col["l_discount"].to_cupy(), dtype=torch.float64)
    sd_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)
    od_t = torch.as_tensor(o_orderdate_days.to_cupy(), dtype=torch.int32)
    sp_t = torch.as_tensor(col["o_shippriority"].to_cupy(), dtype=torch.int32)

    lk_res, rev_res, od_res, sp_res = _query_core(
        lk_t, ep_t, disc_t, sd_t, od_t, sp_t,
        shipdate_lo_days, orderdate_hi_days, int(top_n)
    )

    # Convert orderdate back to datetime64[s] for output
    od_series = cudf.Series(od_res.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame({
        "l_orderkey": lk_res,
        "revenue": rev_res,
        "o_orderdate": od_series,
        "o_shippriority": sp_res,
    })
