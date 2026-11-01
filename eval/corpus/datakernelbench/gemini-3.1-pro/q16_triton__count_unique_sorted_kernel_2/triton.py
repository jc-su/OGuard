import triton
import triton.language as tl

TYPES = {
    "sorted_comp_ptr": "*i64",
    "counts_ptr": "*i64",
    "multiplier": "i32",
    "num_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    sorted_comp_ptr,
    counts_ptr,
    multiplier,
    num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements
    
    # Load current element, using -1 as fallback for out-of-bounds lanes
    curr = tl.load(sorted_comp_ptr + offs, mask=mask, other=-1)
    
    # Load previous element to detect boundaries of identical values
    prev_mask = (offs > 0) & mask
    prev = tl.load(sorted_comp_ptr + offs - 1, mask=prev_mask, other=-1)
    
    # Element is the start of a new unique value if it's the very first element, 
    # or if it is different from the previous element.
    is_new = mask & ((offs == 0) | (curr != prev))
    
    # Decode group_id from the composite key
    group_ids = curr // multiplier
    
    # Prevent out-of-bounds or negative memory access by safely forcing invalid lanes to 0
    safe_group_ids = tl.where(is_new, group_ids, 0)
    
    # Atomic add 1 to the corresponding group's count where a new unique value is found
    ones = tl.full(curr.shape, 1, dtype=tl.int64)
    tl.atomic_add(counts_ptr + safe_group_ids, ones, mask=is_new)
