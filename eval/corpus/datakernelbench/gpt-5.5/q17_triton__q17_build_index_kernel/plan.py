from __future__ import annotations

import os
import tempfile
import importlib.util
import linecache
import types
from typing import Any

import torch
import triton
import cudf


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q17_build_index_kernel(
    part_keys_ptr,
    index_map_ptr,
    n_parts,
    map_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_parts

    key = tl.load(part_keys_ptr + offs, mask=mask, other=-1)
    valid = mask & (key >= 0) & (key < map_size)

    tl.store(index_map_ptr + key, offs.to(tl.int32), mask=valid)


@triton.jit
def _q17_pass1_sum_count_kernel(
    l_partkey_ptr,
    l_quantity_ptr,
    index_map_ptr,
    sum_qty_ptr,
    cnt_qty_ptr,
    n_rows,
    map_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_rows

    partkey = tl.load(l_partkey_ptr + offs, mask=row_mask, other=-1)
    in_range = row_mask & (partkey >= 0) & (partkey < map_size)

    idx = tl.load(index_map_ptr + partkey, mask=in_range, other=-1)
    matched = in_range & (idx >= 0)
    safe_idx = tl.where(idx >= 0, idx, 0)

    qty = tl.load(l_quantity_ptr + offs, mask=matched, other=0.0)
    one = tl.full((BLOCK_SIZE,), 1.0, tl.float64)

    tl.atomic_add(sum_qty_ptr + safe_idx, qty, sem="relaxed", mask=matched)
    tl.atomic_add(cnt_qty_ptr + safe_idx, one, sem="relaxed", mask=matched)


@triton.jit
def _q17_pass2_revenue_kernel(
    l_partkey_ptr,
    l_quantity_ptr,
    l_extendedprice_ptr,
    index_map_ptr,
    sum_qty_ptr,
    cnt_qty_ptr,
    out_ptr,
    n_rows,
    map_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_rows

    partkey = tl.load(l_partkey_ptr + offs, mask=row_mask, other=-1)
    in_range = row_mask & (partkey >= 0) & (partkey < map_size)

    idx = tl.load(index_map_ptr + partkey, mask=in_range, other=-1)
    matched = in_range & (idx >= 0)
    safe_idx = tl.where(idx >= 0, idx, 0)

    qty = tl.load(l_quantity_ptr + offs, mask=matched, other=0.0)
    sum_qty = tl.load(sum_qty_ptr + safe_idx, mask=matched, other=0.0)
    cnt_qty = tl.load(cnt_qty_ptr + safe_idx, mask=matched, other=0.0)

    threshold = (sum_qty / cnt_qty) * 0.2
    keep = matched & (cnt_qty > 0.0) & (qty < threshold)

    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    zero = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    vals = tl.where(keep, price, zero)
    block_sum = tl.sum(vals, axis=0)

    tl.atomic_add(out_ptr, block_sum, sem="relaxed")
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q17_triton_")
    path = os.path.join(tmpdir, "_q17_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)

    spec = importlib.util.spec_from_file_location("_q17_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create import spec for Triton Q17 kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _build_part_index_map(
    part_keys: torch.Tensor,
    block_size: int = 256,
) -> torch.Tensor:
    """
    GPU/Triton helper: builds a dense partkey -> filtered-part ordinal map.

    The selected Q17 part keys are a small dimension after cuDF string filtering.  A dense
    int32 map over the TPC-H part key domain fits easily in H100 L2 and lets the scan kernels
    do one indexed lookup per lineitem instead of a per-row binary search/searchsorted.
    """
    n_parts = int(part_keys.numel())
    if n_parts == 0:
        return torch.empty((0,), dtype=torch.int32, device=part_keys.device)

    max_key = int(torch.max(part_keys).item())
    if max_key < 0:
        return torch.empty((0,), dtype=torch.int32, device=part_keys.device)

    map_size = max_key + 1
    index_map = torch.full((map_size,), -1, dtype=torch.int32, device=part_keys.device)

    grid = (triton.cdiv(n_parts, block_size),)
    _K._q17_build_index_kernel[grid](
        part_keys,
        index_map,
        n_parts,
        map_size,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return index_map


def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC-H Q17 on GPU with Triton kernels.

    Work running on GPU:
      1. Build a dense int32 membership/index map for filtered part keys.
      2. Fused lineitem pass 1: map lookup + conditional quantity load + per-part sum/count.
      3. Fused lineitem pass 2: map lookup + threshold test + conditional price load + global sum.

    This avoids the reference plan's large intermediate boolean/selected tensors and replaces
    per-row searchsorted with one dense map lookup.  The two full lineitem scans are memory-bound,
    coalesced over lineitem columns, and only load quantity/price when predicates require them.
    """
    n_rows = int(l_partkey.numel())
    n_parts = int(part_keys.numel())
    if n_rows == 0 or n_parts == 0:
        return torch.zeros((), dtype=torch.float64, device=l_quantity.device)

    part_keys_i64 = part_keys.to(dtype=torch.int64) if part_keys.dtype != torch.int64 else part_keys
    index_map = _build_part_index_map(part_keys_i64)
    map_size = int(index_map.numel())
    if map_size == 0:
        return torch.zeros((), dtype=torch.float64, device=l_quantity.device)

    sum_qty = torch.zeros((n_parts,), dtype=torch.float64, device=l_quantity.device)
    cnt_qty = torch.zeros((n_parts,), dtype=torch.float64, device=l_quantity.device)
    out = torch.zeros((1,), dtype=torch.float64, device=l_quantity.device)

    block_size = 1024
    grid = (triton.cdiv(n_rows, block_size),)

    _K._q17_pass1_sum_count_kernel[grid](
        l_partkey,
        l_quantity,
        index_map,
        sum_qty,
        cnt_qty,
        n_rows,
        map_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    _K._q17_pass2_revenue_kernel[grid](
        l_partkey,
        l_quantity,
        l_extendedprice,
        index_map,
        sum_qty,
        cnt_qty,
        out,
        n_rows,
        map_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    return out.squeeze(0) / 7.0


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT
            sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM
            lineitem,
            part
        WHERE
            p_partkey = l_partkey
            AND p_brand = 'Brand#23'
            AND p_container = 'MED BOX'
            AND l_quantity < (
                SELECT
                    0.2 * avg(l_quantity)
                FROM
                    lineitem
                WHERE
                    l_partkey = p_partkey
            );

    Description:
        TPC-H Q17. cuDF performs the small string filter on the part table
        (p_brand/p_container).  The bulk lineitem work runs in Triton on GPU:
        a dense filtered-part index map is built, then two fused lineitem scans compute
        per-part average quantity and the final qualifying extendedprice sum.  This is faster
        than the reference eager Torch plan because it avoids materializing selected lineitem
        tensors, avoids scatter/searchsorted temporaries, performs coalesced scans, and loads
        quantity/price conditionally for the low-selectivity part filter.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables containing "lineitem" and "part".
        brand_val (str): Brand filter literal for p_brand.
        container_val (str): Container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    if int(part_keys_t.numel()) == 0:
        return 0.0

    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(),
        dtype=torch.float64,
    )

    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
