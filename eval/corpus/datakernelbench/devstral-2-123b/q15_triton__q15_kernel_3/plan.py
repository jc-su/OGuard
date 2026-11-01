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

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=1)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
    ],
    key=["n_elements"],
    reset_to_zero=["totals_ptr", "present_ptr"],
)
@triton.jit
def _q15_kernel(
    l_suppkey_ptr, l_price_ptr, l_disc_ptr, l_shipdate_ptr,
    suppkey_map_ptr,  # Maps suppkey to dense index
    shipdate_lo, shipdate_hi,
    totals_ptr, present_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load and filter by date
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=shipdate_hi)
    keep = (shipdate >= shipdate_lo) & (shipdate < shipdate_hi) & mask

    # Calculate revenue and get suppkey
    price = tl.load(l_price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(l_disc_ptr + offs, mask=keep, other=0.0)
    revenue = price * (1.0 - disc)
    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)

    # Get dense index from mapping
    dense_idx = tl.load(suppkey_map_ptr + suppkey, mask=keep, other=-1)

    # Atomic operations using dense index
    tl.atomic_add(totals_ptr + dense_idx, revenue, mask=keep & (dense_idx >= 0))
    tl.atomic_add(present_ptr + dense_idx, 1, mask=keep & (dense_idx >= 0))
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
    fusion: bool = True,
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
    Uses a fused Triton kernel with a mapping table to convert sparse suppkey values
    to dense indices for efficient grouping.
    Params:
      tables (dict[str, cudf.DataFrame]): dictionary containing 'lineitem' and 'supplier' tables
      shipdate_lo (str): inclusive lower bound shipdate in YYYY-MM-DD format
      shipdate_hi (str): exclusive upper bound shipdate in YYYY-MM-DD format
      fusion (bool): whether to use the fused Triton path (default True)
    """
    df_line = tables["lineitem"]
    df_supp = tables["supplier"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_suppkey_t = torch.as_tensor(df_line["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_price_t = torch.as_tensor(df_line["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_disc_t = torch.as_tensor(df_line["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate_t = torch.as_tensor(((df_line["l_shipdate"].astype("int64") // 86400)).to_cupy(), dtype=torch.int32)
    s_suppkey_t = torch.as_tensor(df_supp["s_suppkey"].to_cupy(), dtype=torch.int64)

    n_line = l_price_t.numel()
    n_supp = s_suppkey_t.numel()

    if fusion:
        # Create reverse mapping from suppkey to dense index
        max_suppkey = s_suppkey_t.max().item() + 1
        suppkey_map = torch.full((max_suppkey,), -1, dtype=torch.int32, device=l_price_t.device)
        suppkey_map[s_suppkey_t] = torch.arange(n_supp, dtype=torch.int32, device=l_price_t.device)

        totals_t = torch.zeros(n_supp, dtype=torch.float64, device=l_price_t.device)
        present_t = torch.zeros(n_supp, dtype=torch.int32, device=l_price_t.device)

        # Launch kernel
        grid = lambda meta: (triton.cdiv(n_line, meta["BLOCK_SIZE"]),)
        _K._q15_kernel[grid](
            l_suppkey_t, l_price_t, l_disc_t, l_shipdate_t,
            suppkey_map, shipdate_lo_days, shipdate_hi_days,
            totals_t, present_t, n_line,
        )

        # Find suppliers that appeared
        present_mask = present_t > 0

        # Find max revenue and select suppliers
        valid_revenue = torch.where(present_mask, totals_t, torch.tensor(-1.0, dtype=torch.float64, device=totals_t.device))
        max_revenue = valid_revenue.max()
        sel_mask = present_mask & (totals_t == max_revenue)

        # Find indices in original supplier table
        sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

        # Order by s_suppkey
        order = torch.argsort(s_suppkey_t[sel_mask])
        sel_idx = sel_idx[order]
    else:
        # Fallback to reference implementation
        mask = (l_shipdate_t >= shipdate_lo_days) & (l_shipdate_t < shipdate_hi_days)
        suppkey_f = l_suppkey_t[mask]
        revenue_f = l_price_t[mask] * (1.0 - l_disc_t[mask])
        s_supp_sorted, sort_idx = torch.sort(s_suppkey_t)
        pos = torch.searchsorted(s_supp_sorted, suppkey_f)
        valid = (pos < s_supp_sorted.numel()) & (s_supp_sorted[pos.clamp(max=s_supp_sorted.numel()-1)] == suppkey_f)
        pos = pos[valid]
        revenue_f = revenue_f[valid]
        totals_sorted = torch.zeros(n_supp, dtype=torch.float64, device=l_price_t.device)
        totals_sorted.scatter_add_(0, pos, revenue_f)
        totals = torch.empty_like(totals_sorted)
        totals[sort_idx] = totals_sorted
        present_sorted = torch.zeros(n_supp, dtype=torch.bool, device=l_price_t.device)
        present_sorted.scatter_(0, pos, torch.ones_like(pos, dtype=torch.bool))
        present = torch.empty_like(present_sorted)
        present[sort_idx] = present_sorted
        valid_revenue = totals.clone()
        valid_revenue[~present] = -1
        max_revenue = valid_revenue.max()
        sel_mask = present & (totals == max_revenue)
        sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        sel_keys = s_suppkey_t[sel_idx]
        order = torch.argsort(sel_keys)
        sel_idx = sel_idx[order]

    supplier_sel = df_supp.iloc[cudf.Series(sel_idx)].reset_index(drop=True)
    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(totals_t[sel_idx]),
    })
