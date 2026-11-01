import os
import tempfile
import importlib.util
import linecache

import torch
import triton
import triton.language as tl
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source – written to disk at import time.
# ----------------------------------------------------------------------
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
def _q15_agg_kernel(
    revenue_ptr, pos_ptr,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-supplier aggregation using atomic add.
    
    Processes filtered lineitem rows - each row has a pre-computed
    position (pos) into the supplier array.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load revenue and position
    revenue = tl.load(revenue_ptr + offs, mask=mask, other=0.0)
    pos = tl.load(pos_ptr + offs, mask=mask, other=-1)

    # Only accumulate for valid positions (>= 0)
    valid = (pos >= 0) & mask
    safe_pos = tl.where(valid, pos, 0)
    safe_revenue = tl.where(valid, revenue, 0.0)

    tl.atomic_add(out_ptr + safe_pos, safe_revenue)
'''

def _load_kernels():
    """Write the kernel source to a temporary file and load it as a module."""
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


def _launch_q15_agg(
    revenue: torch.Tensor,
    pos: torch.Tensor,
    out: torch.Tensor,
):
    """Launch the aggregation kernel on filtered rows."""
    n = revenue.shape[0]
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q15_agg_kernel[grid](
        revenue,
        pos,
        out,
        n,
    )


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
          WHERE l_shipdate >= DATE '1996-01-01'
            AND l_shipdate < DATE '1996-04-01'
          GROUP BY l_suppkey
      )
      SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
      FROM supplier, revenue
      WHERE s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
      ORDER BY s_suppkey;

    Description:
      TPC‑H Q15 – find the supplier(s) with the highest revenue in a given quarter.
      This implementation filters lineitem rows on the host, then uses a Triton kernel
      for the final per-supplier aggregation, reducing Python overhead.

    Params:
      tables: dict containing 'lineitem' and 'supplier' cuDF DataFrames.
      shipdate_lo: inclusive lower bound of ship date (YYYY‑MM‑DD).
      shipdate_hi: exclusive upper bound of ship date (YYYY‑MM‑DD).
    """
    # ------------------------------------------------------------------
    # 1. Prepare date bounds (int32 epoch days).
    # ------------------------------------------------------------------
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # ------------------------------------------------------------------
    # 2. Load lineitem columns as GPU tensors.
    # ------------------------------------------------------------------
    df_lineitem = tables["lineitem"]
    l_suppkey_t = torch.as_tensor(
        df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64
    )
    l_price_t = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_disc_t = torch.as_tensor(
        df_lineitem["l_discount"].to_cupy(), dtype=torch.float64
    )
    # Convert datetime64[s] to int32 days (exact same conversion as reference)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )

    # ------------------------------------------------------------------
    # 3. Filter lineitem by shipdate (same as reference).
    # ------------------------------------------------------------------
    mask = (l_shipdate_days_t >= shipdate_lo_days) & (l_shipdate_days_t < shipdate_hi_days)
    suppkey_f = l_suppkey_t[mask]
    revenue_f = l_price_t[mask] * (1.0 - l_disc_t[mask])

    # ------------------------------------------------------------------
    # 4. Sort supplier keys and compute searchsorted (same as reference).
    # ------------------------------------------------------------------
    df_supplier = tables["supplier"]
    s_suppkey_t = torch.as_tensor(
        df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64
    )

    sorted_s, sort_idx = torch.sort(s_suppkey_t)
    num_suppliers = s_suppkey_t.shape[0]

    # Compute searchsorted positions for filtered rows
    pos = torch.searchsorted(sorted_s, suppkey_f)

    # Validate positions (same as reference)
    clamped_pos = pos.clamp(max=num_suppliers - 1)
    valid = (pos < num_suppliers) & (sorted_s[clamped_pos] == suppkey_f)
    pos = pos[valid]
    revenue_f = revenue_f[valid]

    # ------------------------------------------------------------------
    # 5. Allocate per-supplier revenue and launch Triton kernel.
    # ------------------------------------------------------------------
    totals_sorted = torch.zeros(num_suppliers, dtype=torch.float64, device=l_suppkey_t.device)

    _launch_q15_agg(
        revenue_f,
        pos,
        totals_sorted,
    )

    # ------------------------------------------------------------------
    # 6. Unsort back to original supplier order (matching reference logic).
    # ------------------------------------------------------------------
    totals_unsorted = torch.empty(num_suppliers, dtype=torch.float64, device=l_suppkey_t.device)
    totals_unsorted[sort_idx] = totals_sorted

    # ------------------------------------------------------------------
    # 7. Find suppliers with the maximum revenue.
    # ------------------------------------------------------------------
    max_rev = totals_unsorted.max()

    # Build present mask (suppliers that had revenue entries)
    present = torch.zeros(num_suppliers, dtype=torch.bool, device=l_suppkey_t.device)
    present_sorted = torch.zeros(num_suppliers, dtype=torch.bool, device=l_suppkey_t.device)
    has_revenue = totals_sorted > 0
    present_sorted[sort_idx] = has_revenue
    present[sort_idx] = present_sorted

    # Select suppliers with max revenue
    sel_mask = present & (totals_unsorted == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

    # Order selected suppliers by s_suppkey (ascending)
    if sel_idx.numel() > 0:
        sel_suppkey = s_suppkey_t[sel_idx]
        order = torch.argsort(sel_suppkey)
        sel_idx = sel_idx[order]

    # ------------------------------------------------------------------
    # 8. Gather string columns from the supplier table and build result.
    # ------------------------------------------------------------------
    sel_idx_series = cudf.Series(sel_idx)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    result = cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(totals_unsorted[sel_idx].cpu()),
    })
    return result
