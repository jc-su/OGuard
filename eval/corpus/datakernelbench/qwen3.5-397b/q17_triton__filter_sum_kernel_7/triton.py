import triton
import triton.language as tl

TYPES = {
    "partkey_ptr": "*i64",
    "quantity_ptr": "*fp64",
    "price_ptr": "*fp64",
    "is_filtered_ptr": "*i32",
    "thresholds_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 4


@triton.jit
def kernel(
    partkey_ptr, quantity_ptr, price_ptr, is_filtered_ptr, thresholds_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    pk = tl.load(partkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    
    filtered = tl.load(is_filtered_ptr + pk, mask=mask, other=0)
    thresh = tl.load(thresholds_ptr + pk, mask=mask, other=0.0)
    
    # Condition: filtered AND qty < 0.2 * avg_qty (stored in thresh)
    cond = (filtered != 0) & (qty < thresh)
    
    contribution = tl.where(cond, price, 0.0)
    tl.atomic_add(out_ptr, tl.sum(contribution))
