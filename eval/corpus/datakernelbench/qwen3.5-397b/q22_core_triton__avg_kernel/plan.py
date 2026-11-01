import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import triton
from typing import Tuple

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096]
        for w in [4, 8]
        for s in [2, 3]
    ],
    key=["n_elements"],
    reset_to_zero=["sum_ptr", "count_ptr"],
)
@triton.jit
def _avg_kernel(
    acctbal_ptr, in_mask_ptr,
    sum_ptr, count_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    in_m = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    
    # Condition: in_mask & bal > 0
    cond = in_m & (bal > 0.0)
    
    # Sum and Count
    local_sum = tl.sum(tl.where(cond, bal, 0.0))
    local_count = tl.sum(cond.to(tl.int64))
    
    tl.atomic_add(sum_ptr, local_sum)
    tl.atomic_add(count_ptr, local_count)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096]
        for w in [4, 8]
        for s in [2, 3]
    ],
    key=["n_elements"],
    reset_to_zero=["group_counts_ptr", "group_sums_ptr"],
)
@triton.jit
def _agg_kernel(
    custkey_ptr, acctbal_ptr, cntrycode_ptr, in_mask_ptr,
    orders_ptr, n_orders,
    avg_val,
    code_map_ptr,
    group_counts_ptr, group_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    ckey = tl.load(custkey_ptr + offs, mask=mask, other=0)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    code = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_m = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    
    # Filter 1 & 2: in_mask & bal > avg_val
    keep = in_m & (bal > avg_val)
    
    # Filter 3: NOT EXISTS (binary search in sorted orders)
    low = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    high = tl.full([BLOCK_SIZE], n_orders, dtype=tl.int64)
    
    # 22 iterations covers up to 4M elements (log2(1.5M) ~ 21)
    for _ in tl.static_range(22):
        mid = (low + high) // 2
        o_mask = mid < n_orders
        # Use a value that won't match valid custkey (>=1) for OOB
        o_val = tl.load(orders_ptr + mid, mask=o_mask, other=-1)
        
        # Update bounds
        # if o_val < ckey: low = mid + 1
        # else: high = mid
        # Treat OOB as +INF (so high = mid)
        is_less = (o_val < ckey) & o_mask
        is_ge = (o_val >= ckey) | (~o_mask)
        
        low = tl.where(is_less, mid + 1, low)
        high = tl.where(is_ge, mid, high)
        
    # Check if found
    check_mask = low < n_orders
    check_val = tl.load(orders_ptr + low, mask=check_mask, other=-1)
    is_in_orders = check_mask & (check_val == ckey)
    
    keep = keep & (~is_in_orders)
    
    # Map code to bin (0..6) using code_map
    # code is int32 (0..99). code_map is size 100.
    bin_idx = tl.load(code_map_ptr + code, mask=mask, other=-1)
    
    final_keep = keep & (bin_idx != -1) & mask
    
    # Safe bin index for atomic add (avoid OOB if bin_idx == -1)
    safe_bin = tl.where(final_keep, bin_idx, 0)
    
    v_count = tl.where(final_keep, 1, 0).to(tl.int64)
    v_sum = tl.where(final_keep, bal, 0.0)
    
    tl.atomic_add(group_counts_ptr + safe_bin, v_count)
    tl.atomic_add(group_sums_ptr + safe_bin, v_sum)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q22_triton_")
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
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized GPU core for TPC-H Q22.
    Uses two fused Triton passes:
    1. Compute avg(c_acctbal) for qualifying customers.
    2. Filter customers (avg, NOT EXISTS) and aggregate by country code.
    """
    device = c_custkey.device
    n = c_custkey.numel()
    
    # Sort o_custkey_unique for binary search (CUDA sort is fast)
    sorted_okeys, _ = torch.sort(o_custkey_unique)
    n_orders = sorted_okeys.numel()
    
    # Prepare code map for 7 target codes, sorted
    # Target codes: 13, 31, 23, 29, 30, 18, 17
    # Sorted: 13, 17, 18, 23, 29, 30, 31
    sorted_targets = [13, 17, 18, 23, 29, 30, 31]
    code_map = torch.full((100,), -1, dtype=torch.int32, device=device)
    for i, code in enumerate(sorted_targets):
        code_map[code] = i
        
    # Pass 1: Compute Avg
    sum_out = torch.zeros(1, dtype=torch.float64, device=device)
    count_out = torch.zeros(1, dtype=torch.int64, device=device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._avg_kernel[grid](
        c_acctbal, in_cntrycode_mask,
        sum_out, count_out,
        n,
    )
    
    total_sum = sum_out.item()
    total_count = count_out.item()
    
    if total_count == 0:
        avg_val = 0.0
    else:
        avg_val = total_sum / total_count
        
    # Pass 2: Aggregate
    group_counts = torch.zeros(7, dtype=torch.int64, device=device)
    group_sums = torch.zeros(7, dtype=torch.float64, device=device)
    
    _K._agg_kernel[grid](
        c_custkey, c_acctbal, cntrycode, in_cntrycode_mask,
        sorted_okeys, n_orders,
        avg_val,
        code_map,
        group_counts, group_sums,
        n,
    )
    
    # Filter out bins with 0 count to match reference (torch.unique on filtered data)
    pos_mask = group_counts > 0
    out_codes = torch.tensor(sorted_targets, dtype=torch.int32, device=device)[pos_mask]
    out_counts = group_counts[pos_mask]
    out_sums = group_sums[pos_mask]
    
    return out_codes, out_counts, out_sums


def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: tuple[str, ...] = ('13', '31', '23', '29', '30', '18', '17'),
) -> cudf.DataFrame:
    """
    SQL: SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal
         FROM (SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
               FROM customer
               WHERE substring(c_phone FROM 1 FOR 2) IN (...)
                 AND c_acctbal > (SELECT avg(c_acctbal) FROM customer
                                  WHERE c_acctbal > 0.00
                                    AND substring(c_phone FROM 1 FOR 2) IN (...))
                 AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
              ) AS custsale
         GROUP BY cntrycode ORDER BY cntrycode;
    Description: TPC-H Q22 - Global sales opportunity query. Identifies customers in
                 target country codes with above-average balances who have placed no orders,
                 grouped and aggregated by country code.
    Params: cntrycodes (tuple[str,...]): target 2-digit country codes.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Derive cntrycode: first 2 chars of c_phone, convert to int (codes are numeric strings)
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")

    # Build in-set mask on GPU
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)

    # Unique o_custkey from orders (for NOT EXISTS)
    o_custkey_unique_series = df_orders["o_custkey"].unique()

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)

    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t, o_custkey_t
    )

    # Format cntrycode back to 2-char zero-padded string
    codes_cudf = cudf.Series(codes_t).astype("int32").astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_t),
        "totacctbal": cudf.Series(totacctbal_t),
    })
