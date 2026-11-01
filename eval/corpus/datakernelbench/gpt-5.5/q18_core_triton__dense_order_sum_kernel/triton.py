import triton
import triton.language as tl

TYPES = {
    "orderkey_ptr": "*i64",
    "quantity_ptr": "*fp64",
    "sums_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    orderkey_ptr,
    quantity_ptr,
    sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keys = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    tl.atomic_add(sums_ptr + keys, qty, mask=mask, sem="relaxed")
