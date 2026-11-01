import triton
import triton.language as tl

TYPES = {
    "o_custkey_ptr": "*i64",
    "bitmap_ptr": "*u8",
    "n_orders": "i32",
    "max_key": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_custkey_ptr,
    bitmap_ptr,
    n_orders,
    max_key,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders
    key = tl.load(o_custkey_ptr + offs, mask=mask, other=-1)
    valid = mask & (key >= 0) & (key <= max_key)
    tl.store(bitmap_ptr + key, 1, mask=valid)
