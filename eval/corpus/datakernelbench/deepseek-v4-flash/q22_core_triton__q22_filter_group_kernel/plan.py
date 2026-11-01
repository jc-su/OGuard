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
def _q22_avg_kernel(
    acctbal_ptr, in_mask_ptr, avg_out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    c_acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    c_in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    
    avg_mask = (c_in_mask == 1) & (c_acctbal > 0.0) & mask
    avg_sum = tl.sum(tl.where(avg_mask, c_acctbal, 0.0))
    avg_count = tl.sum(avg_mask.to(tl.float64))
    
    tl.atomic_add(avg_out_ptr, avg_sum)
    tl.atomic_add(avg_out_ptr + 1, avg_count)

@triton.jit
def _q22_filter_group_kernel(
    custkey_ptr, acctbal_ptr, cntrycode_ptr, in_mask_ptr,
    o_custkey_ptr, o_custkey_n,
    avg_val_ptr, out_numcust_ptr, out_totacctbal_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    c_custkey = tl.load(custkey_ptr + offs, mask=mask, other=-1)
    c_acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    c_cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    c_in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    avg_val = tl.load(avg_val_ptr)
    
    # NOT EXISTS check
    lo = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
    hi = tl.full([BLOCK_SIZE], o_custkey_n, dtype=tl.int64)
    for _ in tl.static_range(32):
        mid = (lo + hi) // 2
        mid_val = tl.load(o_custkey_ptr + mid, mask=mid < o_custkey_n, other=-1)
        lo = tl.where(mid_val < c_custkey, mid + 1, lo)
        hi = tl.where(mid_val < c_custkey, hi, mid)
    found = (lo < o_custkey_n) & (tl.load(o_custkey_ptr + lo, mask=lo < o_custkey_n, other=-1) == c_custkey)
    no_order = ~found
    
    outer_mask = (c_in_mask == 1) & (c_acctbal > avg_val) & no_order & mask
    
    code_map = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
    code_map = tl.where(c_cntrycode == 13, 0, code_map)
    code_map = tl.where(c_cntrycode == 17, 1, code_map)
    code_map = tl.where(c_cntrycode == 18, 2, code_map)
    code_map = tl.where(c_cntrycode == 23, 3, code_map)
    code_map = tl.where(c_cntrycode == 29, 4, code_map)
    code_map = tl.where(c_cntrycode == 30, 5, code_map)
    code_map = tl.where(c_cntrycode == 31, 6, code_map)
    
    valid = outer_mask & (code_map >= 0)
    
    g0_cnt = tl.sum(tl.where(valid & (code_map == 0), 1, 0))
    g1_cnt = tl.sum(tl.where(valid & (code_map == 1), 1, 0))
    g2_cnt = tl.sum(tl.where(valid & (code_map == 2), 1, 0))
    g3_cnt = tl.sum(tl.where(valid & (code_map == 3), 1, 0))
    g4_cnt = tl.sum(tl.where(valid & (code_map == 4), 1, 0))
    g5_cnt = tl.sum(tl.where(valid & (code_map == 5), 1, 0))
    g6_cnt = tl.sum(tl.where(valid & (code_map == 6), 1, 0))
    
    g0_sum = tl.sum(tl.where(valid & (code_map == 0), c_acctbal, 0.0))
    g1_sum = tl.sum(tl.where(valid & (code_map == 1), c_acctbal, 0.0))
    g2_sum = tl.sum(tl.where(valid & (code_map == 2), c_acctbal, 0.0))
    g3_sum = tl.sum(tl.where(valid & (code_map == 3), c_acctbal, 0.0))
    g4_sum = tl.sum(tl.where(valid & (code_map == 4), c_acctbal, 0.0))
    g5_sum = tl.sum(tl.where(valid & (code_map == 5), c_acctbal, 0.0))
    g6_sum = tl.sum(tl.where(valid & (code_map == 6), c_acctbal, 0.0))
    
    tl.atomic_add(out_numcust_ptr + 0, g0_cnt)
    tl.atomic_add(out_numcust_ptr + 1, g1_cnt)
    tl.atomic_add(out_numcust_ptr + 2, g2_cnt)
    tl.atomic_add(out_numcust_ptr + 3, g3_cnt)
    tl.atomic_add(out_numcust_ptr + 4, g4_cnt)
    tl.atomic_add(out_numcust_ptr + 5, g5_cnt)
    tl.atomic_add(out_numcust_ptr + 6, g6_cnt)
    
    tl.atomic_add(out_totacctbal_ptr + 0, g0_sum)
    tl.atomic_add(out_totacctbal_ptr + 1, g1_sum)
    tl.atomic_add(out_totacctbal_ptr + 2, g2_sum)
    tl.atomic_add(out_totacctbal_ptr + 3, g3_sum)
    tl.atomic_add(out_totacctbal_ptr + 4, g4_sum)
    tl.atomic_add(out_totacctbal_ptr + 5, g5_sum)
    tl.atomic_add(out_totacctbal_ptr + 6, g6_sum)
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

def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = c_custkey.numel()
    device = c_custkey.device
    
    sorted_okeys, _ = torch.sort(o_custkey_unique)
    o_custkey_n = sorted_okeys.numel()
    
    avg_out = torch.zeros(2, dtype=torch.float64, device=device)
    BLOCK_SIZE = 4096
    grid_avg = (triton.cdiv(n, BLOCK_SIZE),)
    
    _K._q22_avg_kernel[grid_avg](c_acctbal, in_cntrycode_mask, avg_out, n, BLOCK_SIZE=BLOCK_SIZE)
    
    avg_val = avg_out[0] / avg_out[1]
    
    out_numcust = torch.zeros(7, dtype=torch.int64, device=device)
    out_totacctbal = torch.zeros(7, dtype=torch.float64, device=device)
    avg_val_t = avg_val.unsqueeze(0)
    
    grid_filter = (triton.cdiv(n, BLOCK_SIZE),)
    
    _K._q22_filter_group_kernel[grid_filter](
        c_custkey, c_acctbal, cntrycode, in_cntrycode_mask,
        sorted_okeys, o_custkey_n, avg_val_t,
        out_numcust, out_totacctbal, n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    code_list = torch.tensor([13, 17, 18, 23, 29, 30, 31], dtype=torch.int32, device=device)
    non_zero = out_numcust > 0
    final_codes = code_list[non_zero]
    final_numcust = out_numcust[non_zero]
    final_totacctbal = out_totacctbal[non_zero]
    
    order = torch.argsort(final_codes)
    return final_codes[order], final_numcust[order], final_totacctbal[order]


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

    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)
    o_custkey_unique_series = df_orders["o_custkey"].unique()

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)

    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t, o_custkey_t
    )

    codes_cudf = cudf.Series(codes_t).astype("int32").astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_t),
        "totacctbal": cudf.Series(totacctbal_t),
    })
