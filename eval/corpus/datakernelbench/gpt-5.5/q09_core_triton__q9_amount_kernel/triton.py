import triton
import triton.language as tl

TYPES = {
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "ps_supplycost_ptr": "*fp64",
    "l_quantity_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_extendedprice_ptr,
    l_discount_ptr,
    ps_supplycost_ptr,
    l_quantity_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    extendedprice = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    supplycost = tl.load(ps_supplycost_ptr + offs, mask=mask, other=0.0)
    quantity = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)

    one = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    amount = extendedprice * (one - discount) - supplycost * quantity
    tl.store(out_ptr + offs, amount, mask=mask)
