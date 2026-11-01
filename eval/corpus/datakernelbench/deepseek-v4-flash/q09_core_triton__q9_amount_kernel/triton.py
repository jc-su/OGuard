import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "supplycost_ptr": "*fp64",
    "quantity_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 8192,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr, discount_ptr, supplycost_ptr, quantity_ptr,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    amount = price * (1.0 - discount) - supplycost * quantity
    tl.store(out_ptr + offs, amount, mask=mask)
