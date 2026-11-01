import triton
import triton.language as tl

TYPES = {
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "supplycost_ptr": "*fp64",
    "quantity_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    extendedprice_ptr, discount_ptr, supplycost_ptr, quantity_ptr,
    out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    extendedprice = tl.load(extendedprice_ptr + offs, mask=mask)
    discount = tl.load(discount_ptr + offs, mask=mask)
    supplycost = tl.load(supplycost_ptr + offs, mask=mask)
    quantity = tl.load(quantity_ptr + offs, mask=mask)
    amount = tl.where(mask, (extendedprice * (1.0 - discount)) - (supplycost * quantity), 0.0)
    tl.store(out_ptr + offs, amount)
