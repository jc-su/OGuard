import os
import tempfile
import importlib.util
import linecache
import types

import torch
import triton
import cudf


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _init_map_and_out_kernel(
    map_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(map_ptr + offs, tl.full((BLOCK_SIZE,), -1, tl.int32), mask=mask)
    tl.store(out_ptr, tl.full((), 0.0, tl.float64), mask=pid == 0)


@triton.jit
def _build_map_and_zero_kernel(
    part_keys_ptr,
    map_ptr,
    sum_qty_ptr,
    cnt_qty_ptr,
    n_parts,
    map_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_parts

    keys = tl.load(part_keys_ptr + offs, mask=mask, other=-1)
    valid_key = mask & (keys >= 0) & (keys < map_size)

    tl.store(map_ptr + keys, offs.to(tl.int32), mask=valid_key)
    z = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    tl.store(sum_qty_ptr + offs, z, mask=mask)
    tl.store(cnt_qty_ptr + offs, z, mask=mask)


@triton.jit
def _q17_sum_count_kernel(
    l_partkey_ptr,
    l_quantity_ptr,
    map_ptr,
    sum_qty_ptr,
    cnt_qty_ptr,
    n_lineitems,
    map_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_lineitems

    partkey = tl.load(l_partkey_ptr + offs, mask=mask, other=-1)
    in_range = mask & (partkey >= 0) & (partkey < map_size)

    idx = tl.load(map_ptr + partkey, mask=in_range, other=-1)
    found = in_range & (idx >= 0)
    idx_safe = tl.where(found, idx, 0)

    qty = tl.load(l_quantity_ptr + offs, mask=found, other=0.0)
    one = tl.full((BLOCK_SIZE,), 1.0, tl.float64)

    tl.atomic_add(sum_qty_ptr + idx_safe, qty, sem="relaxed", mask=found)
    tl.atomic_add(cnt_qty_ptr + idx_safe, one, sem="relaxed", mask=found)


@triton.jit
def _q17_final_sum_kernel(
    l_partkey_ptr,
    l_quantity_ptr,
    l_extendedprice_ptr,
    map_ptr,
    sum_qty_ptr,
    cnt_qty_ptr,
    out_ptr,
    n_lineitems,
    map_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_lineitems

    partkey = tl.load(l_partkey_ptr + offs, mask=mask, other=-1)
    in_range = mask & (partkey >= 0) & (partkey < map_size)

    idx = tl.load(map_ptr + partkey, mask=in_range, other=-1)
    found = in_range & (idx >= 0)
    idx_safe = tl.where(found, idx, 0)

    qty = tl.load(l_quantity_ptr + offs, mask=found, other=0.0)
    sum_qty = tl.load(sum_qty_ptr + idx_safe, mask=found, other=0.0)
    cnt_qty = tl.load(cnt_qty_ptr + idx_safe, mask=found, other=1.0)
    cnt_safe = tl.where(found & (cnt_qty > 0.0), cnt_qty, 1.0)

    threshold = (sum_qty / cnt_safe) * 0.2
    keep = found & (qty < threshold)

    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    vals = tl.where(keep, price, tl.full((BLOCK_SIZE,), 0.0, tl.float64))
    block_sum = tl.sum(vals, axis=0)
    tl.atomic_add(out_ptr, block_sum, sem="relaxed")
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q17_triton_")
    path = os.path.join(tmpdir, "_q17_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q17_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q17_map_kernels(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
    block_scan: int = 1024,
    block_init: int = 1024,
    block_build: int = 256,
) -> torch.Tensor:
    """
    Run Q17 numeric work on the GPU with Triton.

    This builds a dense GPU partkey-to-filtered-index map, then performs two fused
    lineitem scans: one for per-part quantity sums/counts and one for the final
    price aggregate. The hot path avoids Torch searchsorted/scatter temporaries
    and conditionally loads quantity/price only for matching filtered parts, which
    is faster for this memory-bound, low-selectivity TPC-H Q17 scan.
    """
    n_lineitems = l_partkey.numel()
    n_parts = part_keys.numel()
    device = l_quantity.device

    if n_lineitems == 0 or n_parts == 0:
        return torch.zeros((), dtype=torch.float64, device=device)

    max_key = int(torch.max(part_keys).item())
    if max_key < 0:
        return torch.zeros((), dtype=torch.float64, device=device)

    map_size = max_key + 1

    part_index_map = torch.empty((map_size,), dtype=torch.int32, device=device)
    sum_qty = torch.empty((n_parts,), dtype=torch.float64, device=device)
    cnt_qty = torch.empty((n_parts,), dtype=torch.float64, device=device)
    out = torch.empty((1,), dtype=torch.float64, device=device)

    grid_init = (triton.cdiv(map_size, block_init),)
    _K._init_map_and_out_kernel[grid_init](
        part_index_map,
        out,
        map_size,
        BLOCK_SIZE=block_init,
        num_warps=8,
    )

    grid_build = (triton.cdiv(n_parts, block_build),)
    _K._build_map_and_zero_kernel[grid_build](
        part_keys,
        part_index_map,
        sum_qty,
        cnt_qty,
        n_parts,
        map_size,
        BLOCK_SIZE=block_build,
        num_warps=4,
    )

    grid_scan = (triton.cdiv(n_lineitems, block_scan),)
    _K._q17_sum_count_kernel[grid_scan](
        l_partkey,
        l_quantity,
        part_index_map,
        sum_qty,
        cnt_qty,
        n_lineitems,
        map_size,
        BLOCK_SIZE=block_scan,
        num_warps=8,
    )

    _K._q17_final_sum_kernel[grid_scan](
        l_partkey,
        l_quantity,
        l_extendedprice,
        part_index_map,
        sum_qty,
        cnt_qty,
        out,
        n_lineitems,
        map_size,
        BLOCK_SIZE=block_scan,
        num_warps=8,
    )

    return out.squeeze(0) / 7.0


def _query_core(
    l_partkey: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    part_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Compute TPC-H Q17 on GPU using Triton kernels.

    cuDF work remains in run_query; this tensor function performs the numeric
    hot path on GPU. It replaces Torch sort/searchsorted/scatter/masking with a
    dense filtered-part lookup table and two fused Triton scans, reducing HBM
    traffic and temporary tensors for the low-selectivity part filter.
    """
    return _launch_q17_map_kernels(l_partkey, l_quantity, l_extendedprice, part_keys)


def run_query(
    tables: dict[str, cudf.DataFrame],
    brand_val: str = "Brand#23",
    container_val: str = "MED BOX",
) -> float:
    """
    SQL:
        SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand = 'Brand#23'
          AND p_container = 'MED BOX'
          AND l_quantity < (SELECT 0.2 * avg(l_quantity)
                            FROM lineitem
                            WHERE l_partkey = p_partkey);
    Description: TPC-H Q17. Computes the small-quantity order revenue average for parts of a
        given brand and container. Correlated subquery is evaluated as a grouped average over
        all lineitems for each filtered partkey.
    Params:
        brand_val (str): brand filter literal for p_brand.
        container_val (str): container filter literal for p_container.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Filter the part table first
    part_mask = (df_part["p_brand"] == brand_val) & (df_part["p_container"] == container_val)
    df_part_f = df_part[part_mask]

    part_keys_t = torch.as_tensor(df_part_f["p_partkey"].to_cupy(), dtype=torch.int64)
    l_partkey_t = torch.as_tensor(df_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(l_partkey_t, l_quantity_t, l_extendedprice_t, part_keys_t)
    return float(res.item())
