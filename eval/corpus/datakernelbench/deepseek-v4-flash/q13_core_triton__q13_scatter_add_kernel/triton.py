import triton
import triton.language as tl

TYPES = {
    "o_custkey_ptr": "*i64",
    "order_counts_ptr": "*i64",
    "n_orders": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 1024,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_custkey_ptr,
    order_counts_ptr,
    n_orders,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders

    o_keys = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    valid = mask & (o_keys > 0)

    # Use tl.atomic_add with block-level offsets
    # Cast to int64 for indexing
    o_keys_int = tl.cast(o_keys, tl.int64)
    ones = tl.where(valid, tl.full((BLOCK_SIZE,), 1, tl.int64), tl.zeros((BLOCK_SIZE,), tl.int64))
    tl.atomic_add(order_counts_ptr + o_keys_int, ones, mask=valid)
