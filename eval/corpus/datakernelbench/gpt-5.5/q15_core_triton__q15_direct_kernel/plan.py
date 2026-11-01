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
def _q15_direct_kernel(
    l_suppkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_days_ptr,
    totals_ptr,
    present_ptr,
    n_elements,
    n_suppliers,
    shipdate_lo_days,
    shipdate_hi_days,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ship_days = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_hi_days)
    keep = mask & (ship_days >= shipdate_lo_days) & (ship_days < shipdate_hi_days)

    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)
    idx = suppkey - 1
    valid = keep & (idx >= 0) & (idx < n_suppliers)

    price = tl.load(l_extendedprice_ptr + offs, mask=valid, other=0.0)
    disc = tl.load(l_discount_ptr + offs, mask=valid, other=0.0)
    revenue = price * (1.0 - disc)

    tl.atomic_add(totals_ptr + idx, revenue, sem="relaxed", mask=valid)
    tl.store(present_ptr + idx, 1, mask=valid)


@triton.jit
def _q15_hash_insert_kernel(
    s_suppkey_ptr,
    hash_keys_ptr,
    hash_vals_ptr,
    table_mask,
):
    idx = tl.program_id(0)
    key = tl.load(s_suppkey_ptr + idx)
    h = (key * 1103515245) & table_mask

    while True:
        old = tl.atomic_cas(hash_keys_ptr + h, -1, key, sem="relaxed")
        if (old == -1) | (old == key):
            tl.store(hash_vals_ptr + h, idx)
            break
        h = (h + 1) & table_mask


@triton.jit
def _q15_hash_agg_kernel(
    l_suppkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_days_ptr,
    hash_keys_ptr,
    hash_vals_ptr,
    totals_ptr,
    present_ptr,
    n_elements,
    table_mask,
    shipdate_lo_days,
    shipdate_hi_days,
    BLOCK_SIZE: tl.constexpr,
    MAX_PROBES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ship_days = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_hi_days)
    keep = mask & (ship_days >= shipdate_lo_days) & (ship_days < shipdate_hi_days)

    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)
    h = (suppkey * 1103515245) & table_mask

    found_idx = tl.full((BLOCK_SIZE,), -1, tl.int64)
    active = keep

    for _ in tl.static_range(0, MAX_PROBES):
        slot_key = tl.load(hash_keys_ptr + h, mask=active, other=-1)
        hit = active & (slot_key == suppkey)
        empty = active & (slot_key == -1)

        slot_val = tl.load(hash_vals_ptr + h, mask=hit, other=-1)
        found_idx = tl.where(hit, slot_val, found_idx)

        active = active & ~(hit | empty)
        h = (h + 1) & table_mask

    valid = keep & (found_idx >= 0)

    price = tl.load(l_extendedprice_ptr + offs, mask=valid, other=0.0)
    disc = tl.load(l_discount_ptr + offs, mask=valid, other=0.0)
    revenue = price * (1.0 - disc)

    tl.atomic_add(totals_ptr + found_idx, revenue, sem="relaxed", mask=valid)
    tl.store(present_ptr + found_idx, 1, mask=valid)
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q15_triton_")
    path = os.path.join(tmpdir, "_q15_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q15_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _next_power_of_2(x: int) -> int:
    """Return the next power of two on the CPU for sizing the small supplier hash table."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _finish_q15_result(
    totals: torch.Tensor,
    present_i8: torch.Tensor,
    s_suppkey: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finish the small supplier-domain result on GPU with PyTorch tensor ops.

    The expensive lineitem scan and revenue aggregation run in fused Triton kernels.
    This helper only processes the 100k-row supplier domain to find the maximum,
    gather ties, and order by supplier key, avoiding the reference's large
    filtered temporaries and searchsorted/scatter work over lineitem rows.
    """
    present = present_i8.to(torch.bool)
    max_rev = totals[present].max()
    sel_mask = present & (totals == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

    sel_keys = s_suppkey[sel_idx]
    order = torch.argsort(sel_keys)
    sel_idx = sel_idx[order]

    return sel_idx, s_suppkey[sel_idx], totals[sel_idx]


def _aggregate_direct_contiguous(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate Q15 revenue with one fused Triton pass for the common TPC-H layout.

    GPU work: Triton scans lineitem once, filters shipdate in registers, conditionally
    loads price/discount/suppkey only for kept rows, computes revenue, and atomically
    accumulates into the supplier array using `l_suppkey - 1`. This is faster than the
    reference because it avoids materializing masks, filtered columns, sort/searchsorted,
    and scatter temporaries over tens of millions of lineitem rows.
    """
    n = int(l_suppkey.numel())
    n_sup = int(s_suppkey.numel())
    totals = torch.zeros(n_sup, dtype=torch.float64, device=s_suppkey.device)
    present = torch.zeros(n_sup, dtype=torch.int8, device=s_suppkey.device)

    if n > 0 and n_sup > 0:
        grid = (triton.cdiv(n, block_size),)
        _K._q15_direct_kernel[grid](
            l_suppkey,
            l_extendedprice,
            l_discount,
            l_shipdate_days,
            totals,
            present,
            n,
            n_sup,
            shipdate_lo_days,
            shipdate_hi_days,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )

    return _finish_q15_result(totals, present, s_suppkey)


def _aggregate_hash_general(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    block_size: int = 1024,
    max_probes: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    General supplier-domain aggregation using Triton-built GPU hash lookup.

    GPU work: Triton first builds a compact open-addressed hash table for the small
    supplier table, then scans lineitem once and performs lookup/aggregation only for
    rows passing the shipdate predicate. This preserves the reference semantics for
    non-contiguous supplier keys while still avoiding large PyTorch filtered
    intermediates and the expensive sort/searchsorted hot path.
    """
    n = int(l_suppkey.numel())
    n_sup = int(s_suppkey.numel())

    totals = torch.zeros(n_sup, dtype=torch.float64, device=s_suppkey.device)
    present = torch.zeros(n_sup, dtype=torch.int8, device=s_suppkey.device)

    if n_sup <= 0:
        return _finish_q15_result(totals, present, s_suppkey)

    table_size = _next_power_of_2(max(16, 4 * n_sup))
    table_mask = table_size - 1

    hash_keys = torch.full((table_size,), -1, dtype=torch.int64, device=s_suppkey.device)
    hash_vals = torch.full((table_size,), -1, dtype=torch.int64, device=s_suppkey.device)

    _K._q15_hash_insert_kernel[(n_sup,)](
        s_suppkey,
        hash_keys,
        hash_vals,
        table_mask,
        num_warps=1,
    )

    if n > 0:
        grid = (triton.cdiv(n, block_size),)
        _K._q15_hash_agg_kernel[grid](
            l_suppkey,
            l_extendedprice,
            l_discount,
            l_shipdate_days,
            hash_keys,
            hash_vals,
            totals,
            present,
            n,
            table_mask,
            shipdate_lo_days,
            shipdate_hi_days,
            BLOCK_SIZE=block_size,
            MAX_PROBES=max_probes,
            num_warps=8,
        )

    return _finish_q15_result(totals, present, s_suppkey)


def _supplier_keys_are_contiguous_1_based(s_suppkey: torch.Tensor) -> bool:
    """
    Check the small supplier tensor layout on GPU.

    TPC-H supplier rows are normally ordered as keys 1..N. Detecting that case lets
    the hot lineitem pass use a direct index calculation instead of hash probes.
    """
    n_sup = int(s_suppkey.numel())
    if n_sup <= 0:
        return False
    expected = torch.arange(1, n_sup + 1, dtype=s_suppkey.dtype, device=s_suppkey.device)
    return bool(torch.equal(s_suppkey, expected))


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
    Compute TPC-H Q15's numeric core on the GPU.

    The hot path is a fused Triton lineitem scan: filter shipdate, compute
    `l_extendedprice * (1 - l_discount)`, aggregate by supplier, and mark present
    suppliers in one pass. For the standard TPC-H contiguous supplier-key layout,
    grouping is a direct array index, minimizing HBM traffic and avoiding the
    reference's large mask/filter/searchsorted/scatter pipeline. A Triton hash-table
    fallback keeps semantics for non-contiguous supplier domains.
    """
    if _supplier_keys_are_contiguous_1_based(s_suppkey):
        return _aggregate_direct_contiguous(
            l_suppkey,
            l_extendedprice,
            l_discount,
            l_shipdate_days,
            s_suppkey,
            shipdate_lo_days,
            shipdate_hi_days,
        )

    return _aggregate_hash_general(
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        s_suppkey,
        shipdate_lo_days,
        shipdate_hi_days,
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
