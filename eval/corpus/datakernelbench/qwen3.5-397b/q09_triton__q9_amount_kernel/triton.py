import triton
import triton.language as tl

TYPES = {
    "ext_price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "supply_cost_ptr": "*fp64",
    "quantity_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 16384
}

NUM_WARPS = 4


@triton.jit
def kernel(
    ext_price_ptr, discount_ptr, supply_cost_ptr, quantity_ptr,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr = 16384,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    ext_price = tl.load(ext_price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    supply_cost = tl.load(supply_cost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    
    amount = ext_price * (1.0 - discount) - supply_cost * quantity
    tl.store(out_ptr + offs, amount, mask=mask)
