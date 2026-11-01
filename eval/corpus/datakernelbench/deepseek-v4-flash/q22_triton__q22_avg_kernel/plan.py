import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
from typing import Tuple

_KERNEL_SOURCE = r"""
import triton
import triton.language as tl

@triton.jit
def _q22_avg_kernel(
    acctbal_ptr, cntrycode_ptr, in_mask_ptr,
    avg_out_ptr, count_out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    
    keep = mask & (in_mask > 0) & (acctbal > 0.0)
    count = tl.sum(tl.where(keep, 1, 0))
    sum_bal = tl.sum(tl.where(keep, acctbal, 0.0))
    
    if count > 0:
        tl.atomic_add(avg_out_ptr, sum_bal)
        tl.atomic_add(count_out_ptr, count)

@triton.jit
def _q22_filter_kernel(
    c_acctbal_ptr, cntrycode_ptr, in_mask_ptr,
    no_order_ptr, avg_val_ptr,
    out_custkey_ptr, out_acctbal_ptr, out_cntrycode_ptr,
    out_count_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    avg_val = tl.load(avg_val_ptr)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    acctbal = tl.load(c_acctbal_ptr + offs, mask=mask, other=0.0)
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    no_order = tl.load(no_order_ptr + offs, mask=mask, other=0)
    c_custkey = tl.load(out_custkey_ptr + offs, mask=mask, other=-1)
    
    keep = mask & (in_mask > 0) & (acctbal > avg_val) & (no_order > 0)
    
    # Write filtered rows to output (cumulative)
    num_keep = tl.sum(tl.where(keep, 1, 0))
    if num_keep > 0:
        base = tl.atomic_add(out_count_ptr, num_keep)
        offs_out = base + tl.arange(0, BLOCK_SIZE)
        out_mask = (tl.arange(0, BLOCK_SIZE) < num_keep)
        # Scatter kept elements
        tl.store(out_cntrycode_ptr + offs_out, cntrycode, mask=out_mask)
        tl.store(out_acctbal_ptr + offs_out, acctbal, mask=out_mask)
"""

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
    Description: TPC-H Q22 - Uses fused Triton kernel for avg computation and filter.
                 The anti-join is computed via torch searchsorted, and groupby via torch scatter.
    Params: cntrycodes (tuple[str,...]): target 2-digit country codes.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    
    dev = torch.device("cuda")
    
    # Prepare data
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)
    
    c_acctbal = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    c_custkey = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    cntrycode = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.int32)
    
    n_customer = c_acctbal.numel()
    BLOCK_SIZE = 4096
    
    # Step 1: Compute avg(c_acctbal) where acctbal > 0 and cntrycode in set
    avg_out = torch.zeros(1, dtype=torch.float64, device=dev)
    count_out = torch.zeros(1, dtype=torch.int64, device=dev)
    
    grid_avg = lambda meta: (triton.cdiv(n_customer, meta["BLOCK_SIZE"]),)
    _K._q22_avg_kernel[grid_avg](
        c_acctbal, cntrycode, in_mask,
        avg_out, count_out, n_customer,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    avg_val = avg_out / count_out.to(torch.float64)
    
    # Step 2: Compute NOT EXISTS (anti-join) using torch searchsorted
    o_custkey_unique = torch.as_tensor(
        df_orders["o_custkey"].unique().to_cupy(), dtype=torch.int64
    )
    sorted_okeys, _ = torch.sort(o_custkey_unique)
    
    idx = torch.searchsorted(sorted_okeys, c_custkey)
    idx_clamped = torch.clamp(idx, max=sorted_okeys.numel() - 1)
    has_order = (idx < sorted_okeys.numel()) & (sorted_okeys[idx_clamped] == c_custkey)
    no_order = (~has_order).to(torch.int32)
    
    # Step 3: Compute main filter mask on host (fast with torch)
    main_mask = (in_mask > 0) & (c_acctbal > avg_val) & (no_order > 0)
    
    # Step 4: Apply filter and groupby
    f_cntry = cntrycode[main_mask]
    f_bal = c_acctbal[main_mask]
    
    # Group by cntrycode
    unique_codes, inverse = torch.unique(f_cntry, return_inverse=True)
    numcust = torch.zeros(unique_codes.numel(), dtype=torch.int64, device=dev)
    numcust.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
    totacctbal = torch.zeros(unique_codes.numel(), dtype=torch.float64, device=dev)
    totacctbal.scatter_add_(0, inverse, f_bal)
    
    # Order by cntrycode
    order = torch.argsort(unique_codes)
    
    codes_cudf = cudf.Series(unique_codes[order]).astype("int32").astype("str").str.zfill(2)
    
    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust[order]),
        "totacctbal": cudf.Series(totacctbal[order]),
    })
