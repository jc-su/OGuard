import triton
import triton.language as tl

TYPES = {
    "l_partkey_ptr": "*i64",
    "l_quantity_ptr": "*fp64",
    "map_ptr": "*i32",
    "sum_qty_ptr": "*fp64",
    "cnt_qty_ptr": "*fp64",
    "n_lineitems": "i32",
    "map_size": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
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
