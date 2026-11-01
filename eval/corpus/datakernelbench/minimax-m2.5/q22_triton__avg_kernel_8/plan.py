import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
from typing import Tuple


# Triton kernels: Q22 - Global Sales Opportunity
# Fusion strategy: one kernel for avg computation, one for final filter+groupby
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Kernel 1: Compute avg(c_acctbal) where c_acctbal > 0 AND cntrycode in target set
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["sum_ptr", "count_ptr"],
)
@triton.jit
def _avg_kernel(
    acctbal_ptr, cntrycode_ptr, in_mask_ptr, 
    sum_ptr, count_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load predicate mask (whether cntrycode is in target set)
    in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    
    # Load cntrycode - use as sentinel if not in set
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    
    # Load acctbal - use 0.0 as sentinel if predicate fails
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    
    # Predicate: in_mask AND acctbal > 0.0
    keep = (in_mask != 0) & (acctbal > 0.0)
    
    # Sum and count where predicate holds
    sum_val = tl.sum(tl.where(keep, acctbal, 0.0))
    count_val = tl.sum(tl.where(keep, 1.0, 0))
    
    tl.atomic_add(sum_ptr, sum_val)
    tl.atomic_add(count_ptr, count_val)


# Kernel 2: Final filter + group by cntrycode
# Group by 7 possible country codes (13, 17, 18, 23, 29, 30, 31)
# Use atomic adds per group
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["numcust_ptr", "totacctbal_ptr"],
)
@triton.jit
def _filter_groupby_kernel(
    acctbal_ptr, cntrycode_ptr, in_mask_ptr, has_order_ptr,
    avg_ptr, numcust_ptr, totacctbal_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Load avg from scalar pointer
    avg_val = tl.load(avg_ptr)
    
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load masks
    in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    has_order = tl.load(has_order_ptr + offs, mask=mask, other=1)  # 1 = has order
    
    # Load acctbal
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    
    # Load cntrycode (int32)
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    
    # Full predicate: in_mask AND acctbal > avg AND NOT has_order
    # has_order == 0 means no order (our mask: 1=has order, 0=no order)
    no_order = (has_order == 0)
    keep = (in_mask != 0) & (acctbal > avg_val) & no_order
    
    # Get group index (cntrycode - 13) / 2 mapping to 0-6
    # Country codes: 13, 17, 18, 23, 29, 30, 31
    # We need to map to 0..6
    # Manual mapping - avoids runtime conditionals
    # 13->0, 17->1, 18->2, 23->3, 29->4, 30->5, 31->6
    gid = -1
    gid = tl.where(cntrycode == 13, 0, gid)
    gid = tl.where(cntrycode == 17, 1, gid)
    gid = tl.where(cntrycode == 18, 2, gid)
    gid = tl.where(cntrycode == 23, 3, gid)
    gid = tl.where(cntrycode == 29, 4, gid)
    gid = tl.where(cntrycode == 30, 5, gid)
    gid = tl.where(cntrycode == 31, 6, gid)
    
    valid_gid = (gid >= 0) & keep
    
    # Count and sum per group
    for i in tl.static_range(7):
        g_keep = valid_gid & (gid == i)
        g_count = tl.sum(tl.where(g_keep, 1, 0))
        g_sum = tl.sum(tl.where(g_keep, acctbal, 0.0))
        tl.atomic_add(numcust_ptr + i, g_count)
        tl.atomic_add(totacctbal_ptr + i, g_sum)
'''


def _load_kernels():
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


def _compute_avg_gpu(
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_mask: torch.Tensor,
) -> float:
    """Compute avg(c_acctbal) where c_acctbal > 0 and cntrycode in target set."""
    n = c_acctbal.numel()
    device = c_acctbal.device
    
    sum_out = torch.zeros(1, dtype=torch.float64, device=device)
    count_out = torch.zeros(1, dtype=torch.int64, device=device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._avg_kernel[grid](
        c_acctbal, cntrycode, in_mask,
        sum_out, count_out, n,
    )
    
    total_sum = sum_out.item()
    total_count = count_out.item()
    
    if total_count == 0:
        return 0.0
    return total_sum / total_count


def _build_has_order_mask(
    c_custkey: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> torch.Tensor:
    """Build has_order mask: 1 if customer has order, 0 otherwise."""
    # Sort unique order custkeys
    sorted_okeys, _ = torch.sort(o_custkey_unique)
    
    # Searchsorted to find potential matches
    idx = torch.searchsorted(sorted_okeys, c_custkey)
    idx_clamped = torch.clamp(idx, max=sorted_okeys.numel() - 1)
    
    # Check if match exists
    has_match = (idx < sorted_okeys.numel()) & (sorted_okeys[idx_clamped] == c_custkey)
    
    # Convert to 0/1 mask (1 = has order)
    return has_match.to(torch.int32)


def _filter_groupby_gpu(
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_mask: torch.Tensor,
    has_order: torch.Tensor,
    avg_val: float,
    num_groups: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter by cntrycode + acctbal > avg + no order, then group by cntrycode."""
    n = c_acctbal.numel()
    device = c_acctbal.device
    
    # Output buffers for 7 groups
    numcust = torch.zeros(num_groups, dtype=torch.int64, device=device)
    totacctbal = torch.zeros(num_groups, dtype=torch.float64, device=device)
    
    # Pass avg as a 1-element tensor so Triton can load from it
    avg_tensor = torch.tensor([avg_val], dtype=torch.float64, device=device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._filter_groupby_kernel[grid](
        c_acctbal, cntrycode, in_mask, has_order,
        avg_tensor, numcust, totacctbal, n,
    )
    
    return numcust, totacctbal


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
                 grouped and aggregated by country code. Uses Triton kernels:
                 - _avg_kernel: fused filter+sum+count for avg computation
                 - _filter_groupby_kernel: fused filter+groupby with atomic adds per group
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

    # Move data to GPU tensors
    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool).to(torch.int32)
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)

    # Step 1: Compute average on GPU using Triton kernel
    avg_val = _compute_avg_gpu(c_acctbal_t, cntrycode_t, in_mask_t)

    # Step 2: Build has_order mask (1 = has order, 0 = no order)
    has_order_t = _build_has_order_mask(c_custkey_t, o_custkey_t)

    # Step 3: Filter + group by on GPU using Triton kernel
    numcust_t, totacctbal_t = _filter_groupby_gpu(
        c_acctbal_t, cntrycode_t, in_mask_t, has_order_t, avg_val
    )

    # Build output - map group indices back to country codes
    # Order: 13, 17, 18, 23, 29, 30, 31 -> sorted
    sorted_codes = sorted(codes_int)
    code_to_idx = {code: idx for idx, code in enumerate(sorted_codes)}
    
    # Reorder outputs by cntrycode
    idx_order = [code_to_idx[c] for c in codes_int]
    numcust_ordered = numcust_t[idx_order]
    totacctbal_ordered = totacctbal_t[idx_order]

    # Format cntrycode back to 2-char zero-padded string
    codes_cudf = cudf.Series(codes_int).astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_ordered),
        "totacctbal": cudf.Series(totacctbal_ordered),
    })
