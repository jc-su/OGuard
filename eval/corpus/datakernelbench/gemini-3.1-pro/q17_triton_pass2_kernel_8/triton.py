import triton
import triton.language as tl

TYPES = {
    "l_partkey_ptr": "*i64",
    "l_quantity_ptr": "*fp64",
    "l_extendedprice_ptr": "*fp64",
    "is_valid_ptr": "*u8",
    "threshold_ptr": "*fp64",
    "out_ptr": "*fp64",
    "max_pk": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_partkey_ptr, l_quantity_ptr, l_extendedprice_ptr,
    is_valid_ptr, threshold_ptr, out_ptr,
    max_pk, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load partkey, default to 0 for out-of-bounds lanes
    pk = tl.load(l_partkey_ptr + offs, mask=mask, other=0)
    
    valid_idx = (pk >= 0) & (pk <= max_pk)
    safe_pk = tl.where(valid_idx, pk, 0)
    
    # Only process partkeys in our valid set
    is_valid = tl.load(is_valid_ptr + safe_pk, mask=mask & valid_idx, other=0)
    keep_part = mask & valid_idx & (is_valid == 1)
    
    # Load quantity and pre-computed threshold
    qty = tl.load(l_quantity_ptr + offs, mask=keep_part, other=0.0).to(tl.float64)
    thresh = tl.load(threshold_ptr + safe_pk, mask=keep_part, other=0.0).to(tl.float64)
    
    # Apply threshold filter logic
    keep = keep_part & (qty < thresh)
    
    # Load extended price only for rows satisfying the subquery
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0).to(tl.float64)
    
    # Block-local reduce followed by global atomic add
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, price, 0.0)))
