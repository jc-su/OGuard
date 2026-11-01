import triton
import triton.language as tl

TYPES = {
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "ps_supplycost_ptr": "*fp64",
    "l_quantity_ptr": "*fp64",
    "amount_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_extendedprice_ptr, l_discount_ptr, ps_supplycost_ptr, l_quantity_ptr,
    amount_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    l_extendedprice = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    l_discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    ps_supplycost = tl.load(ps_supplycost_ptr + offs, mask=mask, other=0.0)
    l_quantity = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    amount = l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity
    tl.store(amount_ptr + offs, amount, mask=mask)
