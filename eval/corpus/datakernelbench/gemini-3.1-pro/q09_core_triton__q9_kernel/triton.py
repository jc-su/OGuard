import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "cost_ptr": "*fp64",
    "qty_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    price_ptr, discount_ptr, cost_ptr, qty_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    price = tl.load(price_ptr + offs, mask=mask)
    discount = tl.load(discount_ptr + offs, mask=mask)
    cost = tl.load(cost_ptr + offs, mask=mask)
    qty = tl.load(qty_ptr + offs, mask=mask)
    
    amount = price * (1.0 - discount) - cost * qty
    tl.store(out_ptr + offs, amount, mask=mask)
