import triton
import triton.language as tl

TYPES = {
    "l_partkey_ptr": "*i64",
    "l_quantity_ptr": "*fp64",
    "l_extendedprice_ptr": "*fp64",
    "index_map_ptr": "*i32",
    "sum_qty_ptr": "*fp64",
    "cnt_qty_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_rows": "i32",
    "map_size": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
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
