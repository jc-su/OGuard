import triton
import triton.language as tl

TYPES = {
    "o_custkey_ptr": "*i64",
    "order_counts_ptr": "*i64",
    "n_orders": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_custkey_ptr,
    order_counts_ptr,
    n_orders,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count orders per customer using atomic add.
    Each order increments its customer's bucket by 1.
    Optimized for H100 with large block sizes.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders
    
    custkey = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    tl.atomic_add(order_counts_ptr + custkey, 1, mask=mask)
