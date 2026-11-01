import triton
import triton.language as tl

TYPES = {
    "partkey_ptr": "*i64",
    "quantity_ptr": "*fp64",
    "is_filtered_ptr": "*i32",
    "sum_qty_ptr": "*fp64",
    "cnt_qty_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 8


@triton.jit
def kernel(
    partkey_ptr, quantity_ptr, is_filtered_ptr,
    sum_qty_ptr, cnt_qty_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    pk = tl.load(partkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    
    # Check if partkey is in filtered set
    filtered = tl.load(is_filtered_ptr + pk, mask=mask, other=0)
    
    # Only accumulate if partkey is filtered
    valid_mask = mask & (filtered != 0)
    
    tl.atomic_add(sum_qty_ptr + pk, qty, mask=valid_mask)
    tl.atomic_add(cnt_qty_ptr + pk, 1.0, mask=valid_mask)
