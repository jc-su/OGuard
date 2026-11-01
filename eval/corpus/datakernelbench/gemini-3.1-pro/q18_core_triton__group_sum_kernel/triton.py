import triton
import triton.language as tl

TYPES = {
    "orderkey_ptr": "*i64",
    "quantity_ptr": "*fp64",
    "sums_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    orderkey_ptr,
    quantity_ptr,
    sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load orderkeys and quantities
    ok = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    
    # Atomic add directly into the pre-allocated index based on the orderkey
    tl.atomic_add(sums_ptr + ok, qty, mask=mask)
