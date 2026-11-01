import triton
import triton.language as tl

TYPES = {
    "o_custkey_ptr": "*i64",
    "counts_ptr": "*i32",
    "n_orders": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_custkey_ptr,
    counts_ptr,
    n_orders,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders
    keys = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    ones = tl.full((BLOCK_SIZE,), 1, dtype=tl.int32)
    tl.atomic_add(counts_ptr + keys, ones, sem="relaxed", mask=mask)
