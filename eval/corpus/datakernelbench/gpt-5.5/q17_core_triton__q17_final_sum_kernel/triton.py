import triton
import triton.language as tl

TYPES = {
    "l_partkey_ptr": "*i64",
    "l_quantity_ptr": "*fp64",
    "l_extendedprice_ptr": "*fp64",
    "map_ptr": "*i32",
    "sum_qty_ptr": "*fp64",
    "cnt_qty_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_lineitems": "i32",
    "map_size": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
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
