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

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["rev_by_suppkey_ptr", "counts_by_suppkey_ptr"],
)
@triton.jit
def _q15_kernel(
    l_suppkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_ptr,
    rev_by_suppkey_ptr,
    counts_by_suppkey_ptr,
    date_lo,
    date_hi,
    max_s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Filter early on shipdate
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Load suppkey only for rows that passed date filter
    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)
    # Ensure suppkey is within the allocated domain bounds
    valid = keep & (suppkey >= 0) & (suppkey <= max_s)
    
    # Load revenue components for valid rows
    price = tl.load(l_extendedprice_ptr + offs, mask=valid, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=valid, other=1.0)
    
    # Compute revenue
    rev = tl.where(valid, price * (1.0 - discount), 0.0)
    
    # Atomic add to the global suppkey buckets
    tl.atomic_add(rev_by_suppkey_ptr + suppkey, rev, mask=valid)
    tl.atomic_add(counts_by_suppkey_ptr + suppkey, 1, mask=valid)
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

def _query_core(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    n_elements = l_suppkey.numel()
    
    # TPC-H keys are 1-based densely packed integers, so max(s_suppkey) provides a safe array size bound
    if s_suppkey.numel() > 0:
        max_s = int(s_suppkey.max().item())
    else:
        max_s = 0
        
    rev_by_suppkey = torch.zeros(max_s + 1, dtype=torch.float64, device=l_suppkey.device)
    counts_by_suppkey = torch.zeros(max_s + 1, dtype=torch.int32, device=l_suppkey.device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    _K._q15_kernel[grid](
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        rev_by_suppkey,
        counts_by_suppkey,
        shipdate_lo_days,
        shipdate_hi_days,
        max_s,
        n_elements
    )
    
    # Extract the aggregates corresponding to the valid supplier table keys
    s_totals = rev_by_suppkey[s_suppkey]
    s_counts = counts_by_suppkey[s_suppkey]
    
    s_present = s_counts > 0
    
    # Guard against empty results returning error on max()
    if not s_present.any():
        empty_idx = torch.empty(0, dtype=torch.int64, device=s_suppkey.device)
        empty_rev = torch.empty(0, dtype=torch.float64, device=s_suppkey.device)
        return empty_idx, empty_idx, empty_rev

    # Find maximum revenue among present suppliers
    max_rev = s_totals[s_present].max()
    
    # Select suppliers equal to max revenue
    sel_mask = s_present & (s_totals == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    
    # Order by s_suppkey
    sel_keys = s_suppkey[sel_idx]
    order = torch.argsort(sel_keys)
    sel_idx = sel_idx[order]
    
    return sel_idx, s_suppkey[sel_idx], s_totals[sel_idx]

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
    Description: TPC-H Q15 — find supplier(s) with the top revenue in a given quarter.
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

    sel_idx_t, sel_keys_t, sel_rev_t = _query_core(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        s_suppkey_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    # Use indices to gather string/phone columns from cuDF supplier table
    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_t),
    })
