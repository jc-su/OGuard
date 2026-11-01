import triton
import triton.language as tl

TYPES = {
    "o_custkey_ptr": "*i64",
    "order_counts_ptr": "*i64",
    "num_orders": "i32",
    "num_customers_plus_one": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 4


@triton.jit
def kernel(
    o_custkey_ptr,
    order_counts_ptr,
    num_orders,
    num_customers_plus_one,
    BLOCK_SIZE: tl.constexpr,
):
    """Count filtered orders per customer using atomic increments."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_orders

    custkey = tl.load(o_custkey_ptr + offs, mask=mask, other=0)
    tl.atomic_add(order_counts_ptr + custkey, 1)
