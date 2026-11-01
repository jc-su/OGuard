import triton
import triton.language as tl

TYPES = {
    "orderkey_ptr": "*i64",
    "qty_ptr": "*fp64",
    "sums_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 1024}
NUM_WARPS = 4


@triton.jit
def kernel(
    orderkey_ptr,          # int64 *
    qty_ptr,               # fp64 *
    sums_ptr,              # fp64 * (size = max_key+1)
    n_elements,            # int64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load only valid lanes; OOB lanes get dummy values (won't affect result)
    order = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)

    # Atomic accumulation per orderkey
    # `order` is int64, valid range is [0, max_key]
    tl.atomic_add(sums_ptr + order, qty)
