import triton
import triton.language as tl

TYPES = {
    "c_custkey_ptr": "*i64",
    "order_counts_ptr": "*i64",
    "c_count_ptr": "*i64",
    "n_customers": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 4


@triton.jit
def kernel(
    c_custkey_ptr,
    order_counts_ptr,
    c_count_ptr,
    n_customers,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Gather order count for each customer from the order_counts array.
    Implements LEFT OUTER JOIN semantics (customers with 0 orders get 0).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers
    
    custkey = tl.load(c_custkey_ptr + offs, mask=mask, other=0)
    count = tl.load(order_counts_ptr + custkey, mask=mask, other=0)
    tl.store(c_count_ptr + offs, count, mask=mask)
