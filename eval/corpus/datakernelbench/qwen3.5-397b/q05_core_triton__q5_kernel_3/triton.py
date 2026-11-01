import triton
import triton.language as tl

TYPES = {
    "nation_id_ptr": "*i64",
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 2048
}

NUM_WARPS = 4


@triton.jit
def kernel(
    nation_id_ptr,
    extendedprice_ptr,
    discount_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    nation_ids = tl.load(nation_id_ptr + offs, mask=mask, other=0)
    prices = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discounts = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    
    revenue = prices * (1.0 - discounts)
    
    tl.atomic_add(out_ptr + nation_ids, revenue, mask=mask)
