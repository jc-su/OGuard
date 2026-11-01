import triton
import triton.language as tl

TYPES = {
    "part_keys_ptr": "*i64",
    "map_ptr": "*i32",
    "sum_qty_ptr": "*fp64",
    "cnt_qty_ptr": "*fp64",
    "n_parts": "i32",
    "map_size": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
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
