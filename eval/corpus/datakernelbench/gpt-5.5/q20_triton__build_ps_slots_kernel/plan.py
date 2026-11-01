from __future__ import annotations

import os
import tempfile
import importlib.util
import linecache
from types import ModuleType
from typing import Any

import cudf
import pandas as pd
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _build_ps_slots_kernel(
    ps_partkey_ptr,
    ps_suppkey_ptr,
    counts_ptr,
    supp_slots_ptr,
    idx_slots_ptr,
    max_slots: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.load(ps_partkey_ptr + row)
    s = tl.load(ps_suppkey_ptr + row)

    slot = tl.atomic_add(counts_ptr + p, 1, sem="relaxed")
    ok = slot < max_slots
    out_pos = p * max_slots + slot

    tl.store(supp_slots_ptr + out_pos, s, mask=ok)
    tl.store(idx_slots_ptr + out_pos, row, mask=ok)


@triton.jit
def _q20_aggregate_lineitem_kernel(
    l_partkey_ptr,
    l_suppkey_ptr,
    l_quantity_ptr,
    l_shipdate_days_ptr,
    counts_ptr,
    supp_slots_ptr,
    idx_slots_ptr,
    sum_qty_ptr,
    n_lineitem,
    n_parts,
    shipdate_lo_days,
    shipdate_hi_days,
    max_slots: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_lineitem

    ship_days = tl.load(l_shipdate_days_ptr + offs, mask=in_bounds, other=shipdate_hi_days)
    date_keep = in_bounds & (ship_days >= shipdate_lo_days) & (ship_days < shipdate_hi_days)

    p = tl.load(l_partkey_ptr + offs, mask=date_keep, other=0)
    s = tl.load(l_suppkey_ptr + offs, mask=date_keep, other=-1)

    valid_part = date_keep & (p >= 0) & (p < n_parts)
    cnt = tl.load(counts_ptr + p, mask=valid_part, other=0)
    has_filtered_part = valid_part & (cnt > 0)

    base = p * max_slots
    found_idx = tl.full((BLOCK_SIZE,), -1, tl.int64)

    for j in tl.static_range(0, max_slots):
        ps_supp = tl.load(supp_slots_ptr + base + j, mask=has_filtered_part, other=-1)
        ps_idx = tl.load(idx_slots_ptr + base + j, mask=has_filtered_part, other=-1)
        hit = has_filtered_part & (s == ps_supp)
        found_idx = tl.where(hit, ps_idx, found_idx)

    matched = found_idx >= 0
    qty = tl.load(l_quantity_ptr + offs, mask=matched, other=0.0)
    tl.atomic_add(sum_qty_ptr + found_idx, qty, sem="relaxed", mask=matched)
'''


def _load_kernels() -> ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q20_triton_")
    path = os.path.join(tmpdir, "_q20_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q20_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import spec for Triton Q20 kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _empty_supplier_result(df_supplier: cudf.DataFrame) -> cudf.DataFrame:
    """Return an empty cuDF result with the SQL projection columns and dtypes."""
    return df_supplier.iloc[:0][["s_name", "s_address"]].reset_index(drop=True)


def _build_partsupp_slots(
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    n_parts: int,
    max_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build dense per-part supplier slots on the GPU with Triton.

    TPC-H partsupp has four suppliers per part at SF10, so a dense partkey -> fixed small
    supplier list avoids the expensive variable-key unique/groupby path.  The build is one
    Triton pass over already prefix-filtered partsupp rows.
    """
    device = ps_partkey.device
    n_slots = int(n_parts) * int(max_slots)
    counts = torch.zeros((n_parts,), dtype=torch.int32, device=device)
    supp_slots = torch.full((n_slots,), -1, dtype=torch.int64, device=device)
    idx_slots = torch.full((n_slots,), -1, dtype=torch.int64, device=device)

    n_ps = int(ps_partkey.numel())
    if n_ps:
        _K._build_ps_slots_kernel[(n_ps,)](
            ps_partkey,
            ps_suppkey,
            counts,
            supp_slots,
            idx_slots,
            max_slots,
            num_warps=1,
        )
    return counts, supp_slots, idx_slots


def _query_core(
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    n_parts: int,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    max_slots: int = 4,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Run the Q20 correlated lineitem aggregate on the GPU with Triton.

    Triton first builds a dense fixed-slot map for the filtered partsupp keys, then performs
    one fused scan of lineitem: date filter, part/supplier membership test, and FP64
    atomic aggregation into one sum per filtered partsupp row.  This replaces the reference
    torch.unique/scatter path with a bandwidth-oriented fused scan and avoids materializing
    a date-filtered lineitem table.
    """
    n_ps = int(ps_partkey.numel())
    sum_qty = torch.zeros((n_ps,), dtype=torch.float64, device=l_quantity.device)
    if n_ps == 0 or int(l_partkey.numel()) == 0:
        return sum_qty

    counts, supp_slots, idx_slots = _build_partsupp_slots(
        ps_partkey=ps_partkey,
        ps_suppkey=ps_suppkey,
        n_parts=n_parts,
        max_slots=max_slots,
    )

    n_lineitem = int(l_partkey.numel())
    grid = lambda meta: (triton.cdiv(n_lineitem, meta["BLOCK_SIZE"]),)
    _K._q20_aggregate_lineitem_kernel[grid](
        l_partkey,
        l_suppkey,
        l_quantity,
        l_shipdate_days,
        counts,
        supp_slots,
        idx_slots,
        sum_qty,
        n_lineitem,
        n_parts,
        shipdate_lo_days,
        shipdate_hi_days,
        max_slots,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return sum_qty


def run_query(
    tables: dict[str, cudf.DataFrame],
    name_prefix: str = "forest",
    nation_name: str = "CANADA",
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
    max_slots: int = 4,
    block_size: int = 256,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT s_name, s_address
        FROM supplier, nation
        WHERE s_suppkey IN (
            SELECT ps_suppkey
            FROM partsupp
            WHERE ps_partkey IN (
                SELECT p_partkey
                FROM part
                WHERE p_name LIKE 'forest%')
              AND ps_availqty > (
                SELECT 0.5 * sum(l_quantity)
                FROM lineitem
                WHERE l_partkey = ps_partkey
                  AND l_suppkey = ps_suppkey
                  AND l_shipdate >= DATE '1994-01-01'
                  AND l_shipdate <  DATE '1995-01-01'))
          AND s_nationkey = n_nationkey
          AND n_name = 'CANADA'
        ORDER BY s_name;

    Description:
        TPC-H Q20.  cuDF handles string prefix filtering, small dimension joins, and final
        ordering/projection.  The bulk correlated subquery over lineitem runs in Triton:
        a fixed-slot GPU map is built for prefix-filtered partsupp rows, then a single fused
        Triton scan applies the shipdate predicate, tests (partkey, suppkey), and accumulates
        FP64 l_quantity sums.  This is faster than the reference Torch plan because it avoids
        torch.unique over concatenated composite keys, avoids materializing filtered lineitem,
        and reduces the hot path to coalesced column reads plus sparse random slot lookups.

    Params:
        tables: Mapping of TPC-H table names to cuDF DataFrames.
        name_prefix: Prefix for p_name LIKE '<prefix>%'.
        nation_name: Nation name predicate.
        shipdate_lo: Inclusive lower shipdate bound.
        shipdate_hi: Exclusive upper shipdate bound.
        max_slots: Fixed supplier slots per part for the dense Triton partsupp map.
        block_size: Triton lineitem scan tile size.
    """
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_partsupp = tables["partsupp"]
    df_part = tables["part"]
    df_lineitem = tables["lineitem"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    part_mask = df_part["p_name"].str.startswith(name_prefix)
    filtered_part = df_part[part_mask][["p_partkey"]]
    if len(filtered_part) == 0:
        return _empty_supplier_result(df_supplier)

    filtered_partsupp = df_partsupp.merge(
        filtered_part,
        left_on="ps_partkey",
        right_on="p_partkey",
        how="inner",
    )[["ps_partkey", "ps_suppkey", "ps_availqty"]].reset_index(drop=True)

    n_ps = len(filtered_partsupp)
    if n_ps == 0:
        return _empty_supplier_result(df_supplier)

    max_partkey = int(df_part["p_partkey"].max())
    n_parts = max_partkey + 1

    ps_partkey_t = torch.as_tensor(filtered_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(filtered_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)

    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy(),
        dtype=torch.int32,
    )

    sum_qty_t = _query_core(
        l_partkey=l_partkey_t,
        l_suppkey=l_suppkey_t,
        l_quantity=l_quantity_t,
        l_shipdate_days=l_shipdate_days_t,
        ps_partkey=ps_partkey_t,
        ps_suppkey=ps_suppkey_t,
        n_parts=n_parts,
        shipdate_lo_days=shipdate_lo_days,
        shipdate_hi_days=shipdate_hi_days,
        max_slots=max_slots,
        block_size=block_size,
    )

    sum_qty_s = cudf.Series(sum_qty_t)
    avail_f = filtered_partsupp["ps_availqty"].astype("float64")
    qualifying_mask = (sum_qty_s > 0.0) & (avail_f > (0.5 * sum_qty_s))
    qualifying_supp = (
        filtered_partsupp.loc[qualifying_mask, ["ps_suppkey"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(qualifying_supp) == 0:
        return _empty_supplier_result(df_supplier)

    nation_filtered = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    if len(nation_filtered) == 0:
        return _empty_supplier_result(df_supplier)

    supplier_in_nation = df_supplier.merge(
        nation_filtered,
        left_on="s_nationkey",
        right_on="n_nationkey",
        how="inner",
    ).reset_index(drop=True)

    if len(supplier_in_nation) == 0:
        return _empty_supplier_result(df_supplier)

    result = supplier_in_nation.merge(
        qualifying_supp,
        left_on="s_suppkey",
        right_on="ps_suppkey",
        how="inner",
    )[["s_name", "s_address"]]

    return result.sort_values("s_name").reset_index(drop=True)
