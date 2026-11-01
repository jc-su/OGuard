import os
import tempfile
import importlib.util
import linecache
import types
from typing import Any

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q15_init_kernel(
    totals_ptr,
    present_ptr,
    count_ptr,
    n_sup: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_sup
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    zi = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
    tl.store(totals_ptr + offs, zf, mask=mask)
    tl.store(present_ptr + offs, zi, mask=mask)
    tl.store(count_ptr, 0, mask=pid == 0)


@triton.jit
def _q15_agg_kernel(
    l_suppkey_ptr,
    l_price_ptr,
    l_discount_ptr,
    l_shipdate_days_ptr,
    sorted_suppkey_ptr,
    totals_ptr,
    present_ptr,
    shipdate_lo_days,
    shipdate_hi_days,
    n_line: tl.constexpr,
    n_sup: tl.constexpr,
    BINARY_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_line

    shipdate = tl.load(l_shipdate_days_ptr + offs, mask=row_mask, other=shipdate_hi_days)
    date_keep = row_mask & (shipdate >= shipdate_lo_days) & (shipdate < shipdate_hi_days)

    supp = tl.load(l_suppkey_ptr + offs, mask=date_keep, other=0)

    lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    hi = lo + n_sup

    for _ in tl.static_range(0, BINARY_STEPS):
        active = date_keep & (lo < hi)
        mid = (lo + hi) // 2
        mid_val = tl.load(sorted_suppkey_ptr + mid, mask=active, other=0)
        go_right = mid_val < supp
        lo = tl.where(active & go_right, mid + 1, lo)
        hi = tl.where(active & (~go_right), mid, hi)

    pos = lo
    pos_in = date_keep & (pos < n_sup)
    found_key = tl.load(sorted_suppkey_ptr + pos, mask=pos_in, other=0)
    found = pos_in & (found_key == supp)

    price = tl.load(l_price_ptr + offs, mask=found, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=found, other=0.0)
    revenue = price * (1.0 - discount)

    tl.atomic_add(totals_ptr + pos, revenue, sem="relaxed", mask=found)
    tl.store(present_ptr + pos, 1, mask=found)


@triton.jit
def _q15_block_max_kernel(
    totals_ptr,
    present_ptr,
    block_max_ptr,
    n_sup: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_sup
    present = tl.load(present_ptr + offs, mask=mask, other=0)
    vals = tl.load(totals_ptr + offs, mask=mask, other=-float("inf"))
    vals = tl.where(mask & (present != 0), vals, -float("inf"))
    m = tl.max(vals, axis=0)
    tl.store(block_max_ptr + pid, m)


@triton.jit
def _q15_final_max_kernel(
    block_max_ptr,
    max_ptr,
    n_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_blocks
    vals = tl.load(block_max_ptr + offs, mask=mask, other=-float("inf"))
    m = tl.max(vals, axis=0)
    tl.store(max_ptr, m)


@triton.jit
def _q15_select_kernel(
    totals_ptr,
    present_ptr,
    sorted_suppkey_ptr,
    max_ptr,
    count_ptr,
    sel_key_ptr,
    sel_rev_ptr,
    n_sup: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_sup

    max_rev = tl.load(max_ptr)
    present = tl.load(present_ptr + offs, mask=mask, other=0)
    vals = tl.load(totals_ptr + offs, mask=mask, other=0.0)

    selected = mask & (present != 0) & (vals == max_rev)
    flags = tl.where(selected, 1, 0)
    n_selected = tl.sum(flags, axis=0)
    prefix = tl.cumsum(flags, axis=0) - 1
    base = tl.atomic_add(count_ptr, n_selected, sem="relaxed")

    keys = tl.load(sorted_suppkey_ptr + offs, mask=selected, other=0)
    out_pos = base + prefix
    tl.store(sel_key_ptr + out_pos, keys, mask=selected)
    tl.store(sel_rev_ptr + out_pos, vals, mask=selected)
'''


def _load_kernels() -> types.ModuleType:
    """
    Write Triton kernels to a real Python file and import them from disk.

    Triton JIT compilation runs from the imported temporary module.  The host
    module keeps cuDF I/O and launch orchestration here, while all GPU kernels
    are in `_KERNEL_SOURCE` to satisfy Triton's source-file requirement.
    """
    tmpdir = tempfile.mkdtemp(prefix="q15_triton_")
    path = os.path.join(tmpdir, "_q15_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q15_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K: types.ModuleType = _load_kernels()


def _next_power_of_2(x: int) -> int:
    """
    Return a host-side power-of-two block size for the final Triton reduction.

    This is CPU scalar setup only; the actual max reduction runs on GPU in a
    Triton kernel.
    """
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _query_core(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Q15 revenue winners.

    GPU/Triton work:
      * initialize per-supplier state;
      * make one fused pass over `lineitem` that filters by shipdate, projects
        `l_extendedprice * (1 - l_discount)`, binary-searches the sorted
        supplier domain, and atomically accumulates revenue;
      * reduce per-supplier totals to the maximum and compact the winning keys.

    Host/PyTorch work is limited to sorting the small supplier key dimension.
    The expensive lineitem scan is memory-bound, so the fused Triton pass avoids
    materializing filter masks, filtered columns, searchsorted outputs, and
    scatter inputs used by the eager reference.
    """
    n_line = int(l_suppkey.numel())
    n_sup = int(s_suppkey.numel())
    device = l_suppkey.device

    if n_sup == 0:
        empty_keys = torch.empty(0, dtype=torch.int64, device=device)
        empty_rev = torch.empty(0, dtype=torch.float64, device=device)
        return empty_keys, empty_rev

    sorted_suppkey, _ = torch.sort(s_suppkey)

    totals = torch.empty(n_sup, dtype=torch.float64, device=device)
    present = torch.empty(n_sup, dtype=torch.uint8, device=device)
    count = torch.empty(1, dtype=torch.int32, device=device)

    init_block = 1024
    init_grid = (triton.cdiv(n_sup, init_block),)
    _K._q15_init_kernel[init_grid](
        totals,
        present,
        count,
        n_sup,
        BLOCK_SIZE=init_block,
        num_warps=4,
    )

    if n_line > 0:
        agg_block = 1024
        binary_steps = max(1, (n_sup - 1).bit_length() + 1)
        agg_grid = (triton.cdiv(n_line, agg_block),)
        _K._q15_agg_kernel[agg_grid](
            l_suppkey,
            l_extendedprice,
            l_discount,
            l_shipdate_days,
            sorted_suppkey,
            totals,
            present,
            shipdate_lo_days,
            shipdate_hi_days,
            n_line,
            n_sup,
            binary_steps,
            BLOCK_SIZE=agg_block,
            num_warps=8,
        )

    reduce_block = 1024
    n_blocks = triton.cdiv(n_sup, reduce_block)
    block_max = torch.empty(n_blocks, dtype=torch.float64, device=device)
    _K._q15_block_max_kernel[(n_blocks,)](
        totals,
        present,
        block_max,
        n_sup,
        BLOCK_SIZE=reduce_block,
        num_warps=4,
    )

    final_block = _next_power_of_2(n_blocks)
    max_rev = torch.empty(1, dtype=torch.float64, device=device)
    _K._q15_final_max_kernel[(1,)](
        block_max,
        max_rev,
        n_blocks,
        BLOCK_SIZE=final_block,
        num_warps=1 if final_block <= 64 else 4,
    )

    sel_keys_buf = torch.empty(n_sup, dtype=torch.int64, device=device)
    sel_rev_buf = torch.empty(n_sup, dtype=torch.float64, device=device)
    select_block = 1024
    select_grid = (triton.cdiv(n_sup, select_block),)
    _K._q15_select_kernel[select_grid](
        totals,
        present,
        sorted_suppkey,
        max_rev,
        count,
        sel_keys_buf,
        sel_rev_buf,
        n_sup,
        BLOCK_SIZE=select_block,
        num_warps=8,
    )

    n_selected = int(count.item())
    return sel_keys_buf[:n_selected], sel_rev_buf[:n_selected]


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
      TPC-H Q15.  cuDF supplies table access, date conversion to int32 epoch
      days, and final string-column assembly.  The hot revenue path runs on GPU
      in fused Triton kernels: one pass over `lineitem` filters, projects, joins
      to the small sorted supplier-key domain by binary search, and accumulates
      per-supplier fp64 totals; follow-up Triton reductions find and compact the
      maximum-revenue supplier(s).  This reduces HBM traffic and temporary
      tensors versus eager mask/filter/searchsorted/scatter operations.

    Params:
      tables (dict[str, cudf.DataFrame]): input TPC-H tables containing
        `lineitem` and `supplier`.
      shipdate_lo (str): inclusive lower shipdate bound.
      shipdate_hi (str): exclusive upper shipdate bound.
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

    sel_keys_t, sel_rev_t = _query_core(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        s_suppkey_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    revenue_df = cudf.DataFrame(
        {
            "s_suppkey": cudf.Series(sel_keys_t),
            "total_revenue": cudf.Series(sel_rev_t),
        }
    )

    supplier_cols = df_supplier[["s_suppkey", "s_name", "s_address", "s_phone"]]
    out = supplier_cols.merge(revenue_df, on="s_suppkey", how="inner")
    out = out.sort_values("s_suppkey").reset_index(drop=True)
    return out[["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]]
