import triton
import triton.language as tl

TYPES = {
    "group_ptr": "*i64",
    "supp_ptr": "*i64",
    "out_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 8


@triton.jit
def kernel(
    group_ptr,
    supp_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Count distinct (group, supp) pairs per group.
    Input must be sorted by (group_id, supp_id).
    Memory-bound optimization: single fused pass over sorted data,
    detecting unique pairs via comparison with predecessor.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    gid = tl.load(group_ptr + offs, mask=mask, other=-1)
    sid = tl.load(supp_ptr + offs, mask=mask, other=-1)
    
    # Load previous element to detect pair boundaries
    prev_offs = offs - 1
    prev_mask = (offs > 0) & mask
    prev_gid = tl.load(group_ptr + prev_offs, mask=prev_mask, other=-1)
    prev_sid = tl.load(supp_ptr + prev_offs, mask=prev_mask, other=-1)
    
    # Mark first occurrence of each (group, supp) pair
    is_first = (offs == 0) | (gid != prev_gid) | (sid != prev_sid)
    is_first = is_first & mask
    
    # Atomically accumulate count per group
    tl.atomic_add(out_ptr + gid, is_first.to(tl.int64), mask=mask)
