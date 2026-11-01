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
    reset_to_zero=["rev_ptr", "present_ptr"],
)
@triton.jit
def _q15_kernel(
    suppkey_ptr, price_ptr, disc_ptr, shipdate_ptr,
    suppkey_to_pos_ptr, rev_ptr, present_ptr,
    date_lo, date_hi, n_suppliers, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate and filter by date range
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    keep_date = (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Load suppkey
    suppkey = tl.load(suppkey_ptr + offs, mask=mask, other=0)
    
    # Lookup position in sorted supplier array using pre-computed mapping
    pos = tl.load(suppkey_to_pos_ptr + suppkey, mask=mask, other=-1)
    valid_supplier = (pos >= 0) & (pos < n_suppliers)
    
    # Combined mask for valid rows
    keep = keep_date & valid_supplier & mask
    
    # Load price and discount only for kept rows
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    
    revenue = price * (1.0 - disc)
    
    # Atomic add to the correct supplier bucket
    tl.atomic_add(rev_ptr + pos, revenue, mask=keep)
    tl.atomic_or(present_ptr + pos, 1, mask=keep)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q15_triton_")
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
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Triton kernel for TPC-H Q15.
    GPU: Filter lineitem by shipdate, compute revenue, aggregate per supplier via atomic_add.
    Host: Pre-compute supplier position mapping, find max revenue, select top suppliers, order by s_suppkey.
    Faster because: single pass over lineitem data with fused filter+project+aggregate.
    """
    n_elements = l_suppkey.numel()
    n_suppliers = s_suppkey.numel()
    
    # Sort supplier keys and create position mapping
    sorted_s, sort_idx = torch.sort(s_suppkey)
    
    # Create lookup table: suppkey -> position in sorted array
    # Max suppkey value determines table size (suppliers are 1-indexed, up to ~100K)
    max_suppkey = sorted_s.max().item()
    suppkey_to_pos = torch.full((max_suppkey + 1,), -1, dtype=torch.int32, device=s_suppkey.device)
    suppkey_to_pos[sorted_s] = torch.arange(n_suppliers, dtype=torch.int32, device=s_suppkey.device)
    
    # Output tensors (in sorted supplier order)
    totals_sorted = torch.zeros(n_suppliers, dtype=torch.float64, device=s_suppkey.device)
    present_sorted = torch.zeros(n_suppliers, dtype=torch.int32, device=s_suppkey.device)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    _K._q15_kernel[grid](
        l_suppkey, l_extendedprice, l_discount, l_shipdate_days,
        suppkey_to_pos, totals_sorted, present_sorted,
        shipdate_lo_days, shipdate_hi_days, n_suppliers, n_elements,
    )
    
    # Unsort back to original supplier order
    present = torch.empty(n_suppliers, dtype=torch.bool, device=s_suppkey.device)
    present[sort_idx] = present_sorted.to(torch.bool)
    
    totals = torch.empty(n_suppliers, dtype=torch.float64, device=s_suppkey.device)
    totals[sort_idx] = totals_sorted
    
    # Find max revenue among present suppliers
    present_rev = totals[present]
    if present_rev.numel() == 0:
        sel_idx = torch.empty(0, dtype=torch.int64, device=s_suppkey.device)
        return sel_idx, torch.empty(0, dtype=torch.int64, device=s_suppkey.device), torch.empty(0, dtype=torch.float64, device=s_suppkey.device)
    
    max_rev = present_rev.max()
    
    # Select suppliers with max revenue
    sel_mask = present & (totals == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    
    # Order by s_suppkey
    if sel_idx.numel() > 0:
        sel_keys = s_suppkey[sel_idx]
        order = torch.argsort(sel_keys)
        sel_idx = sel_idx[order]
    
    return sel_idx, s_suppkey[sel_idx], totals[sel_idx]


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

    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_t),
    })
