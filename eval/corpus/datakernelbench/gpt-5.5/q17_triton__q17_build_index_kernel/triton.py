import triton
import triton.language as tl

TYPES = {
    "part_keys_ptr": "*i64",
    "index_map_ptr": "*i32",
    "n_parts": "i32",
    "map_size": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
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
