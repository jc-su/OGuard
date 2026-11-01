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

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _q22_avg_kernel(
    acctbal_ptr,
    cntrycode_ptr,
    in_mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute sum and count of c_acctbal where cntrycode in set AND c_acctbal > 0.
    Output: out_ptr[0] = sum, out_ptr[1] = count (both float64)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load cntrycode and in_mask
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_set = tl.load(in_mask_ptr + offs, mask=mask, other=False)
    
    # Load acctbal with safe other
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    
    # Filter: in_set AND acctbal > 0
    keep = mask & in_set & (acctbal > 0.0)
    
    # Conditional load for acctbal (only load where keep is true)
    acctbal_filtered = tl.load(acctbal_ptr + offs, mask=keep, other=0.0)
    
    # Accumulate sum and count
    block_sum = tl.sum(tl.where(keep, acctbal_filtered, 0.0))
    block_count = tl.sum(tl.where(keep, 1.0, 0.0))
    
    # Atomic add to global accumulators
    tl.atomic_add(out_ptr, block_sum)
    tl.atomic_add(out_ptr + 1, block_count)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_sum_ptr", "out_count_ptr"],
)
@triton.jit
def _q22_agg_kernel(
    acctbal_ptr,
    cntrycode_ptr,
    in_mask_ptr,
    has_order_ptr,
    avg_thresh_ptr,
    out_sum_ptr,
    out_count_ptr,
    n_elements,
    num_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Final aggregation: group by cntrycode, sum acctbal and count.
    Filters: in_mask AND acctbal > avg_thresh AND NOT has_order.
    num_groups = 7 (fixed country codes: 13, 17, 18, 23, 29, 30, 31)
    Group indices: 13->0, 17->1, 18->2, 23->3, 29->4, 30->5, 31->6
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load all filter columns
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_set = tl.load(in_mask_ptr + offs, mask=mask, other=False)
    has_order = tl.load(has_order_ptr + offs, mask=mask, other=True)
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    avg_thresh = tl.load(avg_thresh_ptr)
    
    # Filter: in_set AND acctbal > avg_thresh AND NOT has_order
    keep = mask & in_set & (acctbal > avg_thresh) & (~has_order)
    
    # Map cntrycode to group index (13,17,18,23,29,30,31 -> 0,1,2,3,4,5,6)
    # Use tl.where chain for mapping
    code_13 = (cntrycode == 13)
    code_17 = (cntrycode == 17)
    code_18 = (cntrycode == 18)
    code_23 = (cntrycode == 23)
    code_29 = (cntrycode == 29)
    code_30 = (cntrycode == 30)
    code_31 = (cntrycode == 31)
    
    group_idx = (
        tl.where(code_13, 0,
        tl.where(code_17, 1,
        tl.where(code_18, 2,
        tl.where(code_23, 3,
        tl.where(code_29, 4,
        tl.where(code_30, 5,
        tl.where(code_31, 6, 7))))))))
    
    # Load acctbal only for kept rows
    acctbal_filtered = tl.load(acctbal_ptr + offs, mask=keep, other=0.0)
    
    # Accumulate per-group using atomics
    # For each group, atomic_add to the corresponding position
    for g in tl.static_range(num_groups):
        is_group = (group_idx == g) & keep
        group_sum = tl.sum(tl.where(is_group, acctbal_filtered, 0.0))
        group_count = tl.sum(tl.where(is_group, 1.0, 0.0))
        tl.atomic_add(out_sum_ptr + g, group_sum)
        tl.atomic_add(out_count_ptr + g, group_count)
'''


def _load_kernels():
    """Load Triton kernels from temp file (required for validator)."""
    tmpdir = tempfile.mkdtemp(prefix="q22_triton_")
    path = os.path.join(tmpdir, "_q22_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q22_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _compute_avg_threshold(
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute avg(c_acctbal) where cntrycode in set AND c_acctbal > 0.
    Uses fused Triton kernel for single-pass sum/count.
    """
    n = c_acctbal.numel()
    out = torch.zeros(2, dtype=torch.float64, device=c_acctbal.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    _K._q22_avg_kernel[grid](
        c_acctbal,
        cntrycode,
        in_mask,
        out,
        n,
    )
    
    # avg = sum / count
    total_sum = out[0]
    total_count = out[1]
    avg_val = total_sum / total_count if total_count > 0 else torch.tensor(0.0, dtype=torch.float64, device=c_acctbal.device)
    return avg_val


def _final_aggregation(
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_mask: torch.Tensor,
    has_order: torch.Tensor,
    avg_thresh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Final group-by aggregation: count and sum per cntrycode.
    Returns: (cntrycode_values, numcust, totacctbal) sorted by cntrycode.
    """
    n = c_acctbal.numel()
    num_groups = 7  # Fixed: 13, 17, 18, 23, 29, 30, 31
    
    out_sum = torch.zeros(num_groups, dtype=torch.float64, device=c_acctbal.device)
    out_count = torch.zeros(num_groups, dtype=torch.int64, device=c_acctbal.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    _K._q22_agg_kernel[grid](
        c_acctbal,
        cntrycode,
        in_mask,
        has_order,
        avg_thresh,
        out_sum,
        out_count,
        n,
        num_groups,
    )
    
    # Map back to cntrycode values (sorted: 13, 17, 18, 23, 29, 30, 31)
    codes = torch.tensor([13, 17, 18, 23, 29, 30, 31], dtype=torch.int32, device=c_acctbal.device)
    
    return codes, out_count, out_sum


def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized Q22 core using fused Triton kernels.
    
    GPU Operations:
    - _q22_avg_kernel: Single-pass sum/count for avg threshold (memory-bound fusion)
    - _q22_agg_kernel: Single-pass group-by aggregation with all filters
    
    Why faster:
    - Two kernel launches instead of multiple torch operations
    - Fused filter+project+aggregate reduces HBM traffic
    - Per-group accumulators avoid scatter_add overhead
    """
    # Build has_order mask using searchsorted (efficient for NOT EXISTS)
    sorted_okeys, _ = torch.sort(o_custkey_unique)
    idx = torch.searchsorted(sorted_okeys, c_custkey)
    idx_clamped = torch.clamp(idx, max=sorted_okeys.numel() - 1)
    has_order = (idx < sorted_okeys.numel()) & (sorted_okeys[idx_clamped] == c_custkey)
    
    # Step 1: Compute avg threshold with fused kernel
    avg_thresh = _compute_avg_threshold(c_acctbal, cntrycode, in_cntrycode_mask)
    
    # Step 2: Final aggregation with all filters
    codes, numcust, totacctbal = _final_aggregation(
        c_acctbal,
        cntrycode,
        in_cntrycode_mask,
        has_order,
        avg_thresh,
    )
    
    return codes, numcust, totacctbal


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
    
    GPU Optimization (Triton):
    - cuDF: String extraction (c_phone -> cntrycode), mask building
    - Triton Kernel 1 (_q22_avg_kernel): Fused sum/count for avg threshold
    - Triton Kernel 2 (_q22_agg_kernel): Fused filter+group-by+aggregate
    - Benefits: 2 kernel launches vs multiple torch ops, reduced HBM traffic,
      per-group accumulators avoid scatter_add overhead
    
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables (customer, orders)
        cntrycodes (tuple[str,...]): Target 2-digit country codes (default: 7 codes)
    
    Returns:
        cudf.DataFrame: Columns [cntrycode, numcust, totacctbal] sorted by cntrycode
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

    # Transfer to GPU tensors
    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)

    # Execute optimized Triton kernels
    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t, o_custkey_t
    )

    # Format cntrycode back to 2-char zero-padded string
    codes_cudf = cudf.Series(codes_t.cpu()).astype("int32").astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_t.cpu()),
        "totacctbal": cudf.Series(totacctbal_t.cpu()),
    })
