from __future__ import annotations

import os
import tempfile
import importlib.util
import linecache
from typing import Any

import cudf
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _zero3_i32_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    tl.store(a_ptr + offs, z, mask=mask)
    tl.store(b_ptr + offs, z, mask=mask)
    tl.store(c_ptr + offs, z, mask=mask)


@triton.jit
def _build_part_maps_kernel(
    p_partkey_ptr,
    p_brand_id_ptr,
    p_container_grp_ptr,
    p_size_ptr,
    brand_map_ptr,
    container_map_ptr,
    size_map_ptr,
    map_size,
    n_parts,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_parts

    key = tl.load(p_partkey_ptr + offs, mask=mask, other=-1)
    valid = mask & (key >= 0) & (key < map_size)

    brand_id = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    container_grp = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    p_size = tl.load(p_size_ptr + offs, mask=mask, other=0)

    tl.store(brand_map_ptr + key, brand_id, mask=valid)
    tl.store(container_map_ptr + key, container_grp, mask=valid)
    tl.store(size_map_ptr + key, p_size, mask=valid)


@triton.jit
def _q19_lineitem_kernel(
    l_quantity_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_partkey_ptr,
    ship_ok_ptr,
    brand_map_ptr,
    container_map_ptr,
    size_map_ptr,
    max_partkey,
    brand12_id,
    brand23_id,
    brand34_id,
    sm_grp,
    md_grp,
    lg_grp,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_elements

    ship_ok = tl.load(ship_ok_ptr + offs, mask=row_mask, other=0)
    ship_mask = row_mask & (ship_ok != 0)

    qty = tl.load(l_quantity_ptr + offs, mask=ship_mask, other=0.0)
    qty_candidate = ship_mask & (qty >= 1.0) & (qty <= 30.0)

    partkey = tl.load(l_partkey_ptr + offs, mask=qty_candidate, other=-1)
    part_mask = qty_candidate & (partkey >= 0) & (partkey <= max_partkey)

    brand_id = tl.load(brand_map_ptr + partkey, mask=part_mask, other=0)
    container_grp = tl.load(container_map_ptr + partkey, mask=part_mask, other=0)
    p_size = tl.load(size_map_ptr + partkey, mask=part_mask, other=0)

    cond1 = (
        (brand_id == brand12_id)
        & (container_grp == sm_grp)
        & (qty >= 1.0)
        & (qty <= 11.0)
        & (p_size >= 1)
        & (p_size <= 5)
    )
    cond2 = (
        (brand_id == brand23_id)
        & (container_grp == md_grp)
        & (qty >= 10.0)
        & (qty <= 20.0)
        & (p_size >= 1)
        & (p_size <= 10)
    )
    cond3 = (
        (brand_id == brand34_id)
        & (container_grp == lg_grp)
        & (qty >= 20.0)
        & (qty <= 30.0)
        & (p_size >= 1)
        & (p_size <= 15)
    )

    keep = part_mask & (cond1 | cond2 | cond3)

    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)

    one = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    zero = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    contrib = tl.where(keep, price * (one - discount), zero)

    tl.atomic_add(out_ptr, tl.sum(contrib))
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q19_triton_")
    path = os.path.join(tmpdir, "_q19_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q19_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _encode_part_columns(
    df_part: cudf.DataFrame,
    brand1: str,
    brand2: str,
    brand3: str,
) -> tuple[cudf.Series, cudf.Series, cudf.Series]:
    """
    Run cuDF string comparisons on the small part table and produce numeric columns.

    cuDF handles string predicates; the resulting compact integer codes are consumed by
    Triton so the large lineitem scan does not materialize joined string columns.
    """
    p_brand = df_part["p_brand"]
    p_container = df_part["p_container"]

    brand_id = (
        (p_brand == brand1).astype("int32") * 12
        + (p_brand == brand2).astype("int32") * 23
        + (p_brand == brand3).astype("int32") * 34
    )

    sm = p_container.isin(["SM CASE", "SM BOX", "SM PACK", "SM PKG"]).astype("int32")
    md = p_container.isin(["MED BAG", "MED BOX", "MED PKG", "MED PACK"]).astype("int32")
    lg = p_container.isin(["LG CASE", "LG BOX", "LG PACK", "LG PKG"]).astype("int32")
    container_grp = sm * 1 + md * 2 + lg * 3

    p_size = df_part["p_size"].astype("int32")
    return brand_id.astype("int32"), container_grp.astype("int32"), p_size


def _build_part_maps(
    df_part: cudf.DataFrame,
    brand1: str,
    brand2: str,
    brand3: str,
    map_block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build dense GPU lookup maps keyed by p_partkey.

    cuDF encodes string part attributes; Triton initializes and fills dense int32 maps.
    This replaces a large cuDF hash join with coalesced lineitem scanning plus random
    lookups into a small part dimension that fits comfortably in GPU cache.
    """
    n_parts = len(df_part)
    if n_parts == 0:
        device = torch.device("cuda")
        brand_map = torch.zeros(1, dtype=torch.int32, device=device)
        container_map = torch.zeros(1, dtype=torch.int32, device=device)
        size_map = torch.zeros(1, dtype=torch.int32, device=device)
        return brand_map, container_map, size_map, 0

    brand_id_s, container_grp_s, p_size_s = _encode_part_columns(df_part, brand1, brand2, brand3)

    max_partkey = int(df_part["p_partkey"].max())
    map_size = max_partkey + 1

    p_partkey_t = torch.as_tensor(df_part["p_partkey"].to_cupy(), dtype=torch.int64)
    brand_id_t = torch.as_tensor(brand_id_s.to_cupy(), dtype=torch.int32)
    container_grp_t = torch.as_tensor(container_grp_s.to_cupy(), dtype=torch.int32)
    p_size_t = torch.as_tensor(p_size_s.to_cupy(), dtype=torch.int32)

    device = p_partkey_t.device
    brand_map = torch.empty(map_size, dtype=torch.int32, device=device)
    container_map = torch.empty(map_size, dtype=torch.int32, device=device)
    size_map = torch.empty(map_size, dtype=torch.int32, device=device)

    zero_grid = (triton.cdiv(map_size, map_block_size),)
    _K._zero3_i32_kernel[zero_grid](
        brand_map,
        container_map,
        size_map,
        map_size,
        BLOCK_SIZE=map_block_size,
    )

    build_grid = (triton.cdiv(n_parts, map_block_size),)
    _K._build_part_maps_kernel[build_grid](
        p_partkey_t,
        brand_id_t,
        container_grp_t,
        p_size_t,
        brand_map,
        container_map,
        size_map,
        map_size,
        n_parts,
        BLOCK_SIZE=map_block_size,
    )

    return brand_map, container_map, size_map, max_partkey


def _query_core(
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_partkey: torch.Tensor,
    ship_ok: torch.Tensor,
    brand_map: torch.Tensor,
    container_map: torch.Tensor,
    size_map: torch.Tensor,
    max_partkey: int,
    brand12_id: int = 12,
    brand23_id: int = 23,
    brand34_id: int = 34,
    sm_grp: int = 1,
    md_grp: int = 2,
    lg_grp: int = 3,
    block_size: int = 2048,
) -> torch.Tensor:
    """
    Run the hot path in one fused Triton lineitem scan.

    The kernel applies ship, quantity, part-brand/container/size predicates, conditionally
    loads price/discount only for matching rows, and performs an FP64 global reduction.
    This is faster for the memory-bound scan because it avoids materializing a joined table
    and avoids multiple temporary boolean/revenue columns.
    """
    n = l_quantity.numel()
    out = torch.zeros(1, dtype=torch.float64, device=l_quantity.device)
    if n == 0:
        return out.squeeze(0)

    grid = (triton.cdiv(n, block_size),)
    _K._q19_lineitem_kernel[grid](
        l_quantity,
        l_extendedprice,
        l_discount,
        l_partkey,
        ship_ok,
        brand_map,
        container_map,
        size_map,
        max_partkey,
        brand12_id,
        brand23_id,
        brand34_id,
        sm_grp,
        md_grp,
        lg_grp,
        out,
        n,
        BLOCK_SIZE=block_size,
    )
    return out.squeeze(0)


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand1: str = "Brand#12",
    brand2: str = "Brand#23",
    brand3: str = "Brand#34",
    shipinstruct: str = "DELIVER IN PERSON",
    block_size: int = 2048,
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue
        FROM lineitem, part
        WHERE (p_partkey = l_partkey
          AND p_brand = 'Brand#12'
          AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
          AND l_quantity >= 1 AND l_quantity <= 11
          AND p_size BETWEEN 1 AND 5
          AND l_shipmode IN ('AIR', 'AIR REG')
          AND l_shipinstruct = 'DELIVER IN PERSON')
        OR (p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
          AND l_quantity >= 10 AND l_quantity <= 20
          AND p_size BETWEEN 1 AND 10
          AND l_shipmode IN ('AIR', 'AIR REG')
          AND l_shipinstruct = 'DELIVER IN PERSON')
        OR (p_partkey = l_partkey
          AND p_brand = 'Brand#34'
          AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
          AND l_quantity >= 20 AND l_quantity <= 30
          AND p_size BETWEEN 1 AND 15
          AND l_shipmode IN ('AIR', 'AIR REG')
          AND l_shipinstruct = 'DELIVER IN PERSON').

    Description:
        TPC-H Q19 discounted revenue. cuDF performs unavoidable string work: lineitem
        ship-mode/instruction mask creation and part brand/container encoding. Triton
        then builds dense part lookup maps and executes a single fused GPU pass over
        lineitem for join lookup, predicates, projection, and FP64 aggregation. This
        avoids the reference plan's large cuDF join, joined string encodings, eager
        boolean temporaries, and separate revenue materialization.

    Params:
        tables: Dictionary containing cuDF DataFrames "lineitem" and "part".
        brand1, brand2, brand3: Brand literals for the three SQL disjuncts.
        shipinstruct: Ship-instruction literal.
        block_size: Triton scan tile size; defaults to 2048 for H100-friendly occupancy.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    brand_map, container_map, size_map, max_partkey = _build_part_maps(
        df_part,
        brand1,
        brand2,
        brand3,
    )

    ship_ok_s = (
        df_lineitem["l_shipmode"].isin(["AIR", "AIR REG"])
        & (df_lineitem["l_shipinstruct"] == shipinstruct)
    ).astype("int8")

    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    ship_ok_t = torch.as_tensor(ship_ok_s.to_cupy(), dtype=torch.int8)

    result = _query_core(
        l_quantity_t,
        l_extendedprice_t,
        l_discount_t,
        l_partkey_t,
        ship_ok_t,
        brand_map,
        container_map,
        size_map,
        max_partkey,
        12,
        23,
        34,
        1,
        2,
        3,
        block_size,
    )
    return float(result.item())
