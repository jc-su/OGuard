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
def _q15_kernel(
    shipdate_ptr,
    price_ptr,
    disc_ptr,
    supp_ptr,
    totals_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 8192,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    supp = tl.load(supp_ptr + offs, mask=mask, other=0)
    
    date_ok = (shipdate >= date_lo) & (shipdate < date_hi)
    valid = date_ok & (supp > 0)
    
    revenue = price * (1.0 - disc)
    rev_val = tl.where(valid, revenue, 0.0)
    
    tl.atomic_add(totals_ptr + supp, rev_val)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q15_triton_")
    path = os.path.join(tmpdir, "_q15_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q15_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1996-01-01",
    shipdate_hi: str = "1996-04-01",
) -> cudf.DataFrame:
    """
    SQL:
      WITH revenue AS (
        SELECT l_suppkey AS supplier_no,
               SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
        GROUP BY l_suppkey
      )
      SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
      FROM supplier, revenue
      WHERE s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
      ORDER BY s_suppkey;
    Description: 
      TPC-H Q15 with fused Triton kernel. Direct indexing by suppkey (1-based).
      No key_map overhead - TPC-H supplier keys are sequential 1..N.
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_price_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_disc_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    
    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)
    n_sup = s_suppkey_t.numel()
    
    # TPC-H supplier keys are 1-based sequential (1..100000 for SF10)
    # Use direct indexing: totals[suppkey] = revenue
    totals = torch.zeros(n_sup + 1, dtype=torch.float64, device=s_suppkey_t.device)
    
    n_elements = l_suppkey_t.numel()
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _K._q15_kernel[grid](
        l_shipdate_days_t,
        l_price_t,
        l_disc_t,
        l_suppkey_t,
        totals,
        shipdate_lo_days,
        shipdate_hi_days,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Extract revenues for actual suppliers (indices 1..n_sup)
    totals_sup = totals[1:n_sup+1]
    
    # Sort supplier keys to get correct ordering
    sorted_keys, sort_indices = torch.sort(s_suppkey_t)
    totals_sorted = totals_sup[sort_indices]
    
    max_rev = totals_sorted.max()
    
    if max_rev <= 0.0:
        return cudf.DataFrame({
            "s_suppkey": cudf.Series([], dtype="int64"),
            "s_name": cudf.Series([], dtype="object"),
            "s_address": cudf.Series([], dtype="object"),
            "s_phone": cudf.Series([], dtype="object"),
            "total_revenue": cudf.Series([], dtype="float64"),
        })

    sel_mask_sorted = (totals_sorted == max_rev) & (totals_sorted > 0.0)
    sel_idx_sorted = torch.nonzero(sel_mask_sorted, as_tuple=False).squeeze(1)
    
    sel_keys = sorted_keys[sel_idx_sorted]
    sel_orig_idx = sort_indices[sel_idx_sorted]
    sel_orig_idx_series = cudf.Series(sel_orig_idx)
    supplier_sel = df_supplier.iloc[sel_orig_idx_series].reset_index(drop=True)
    
    return cudf.DataFrame({
        "s_suppkey": cudf.Series(sel_keys.cpu().numpy()),
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(totals_sorted[sel_idx_sorted].cpu().numpy()),
    })
