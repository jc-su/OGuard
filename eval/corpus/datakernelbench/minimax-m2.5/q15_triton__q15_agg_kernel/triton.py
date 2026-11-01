import triton
import triton.language as tl

TYPES = {
    "revenue_ptr": "*fp64",
    "pos_ptr": "*i64",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    revenue_ptr, pos_ptr,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-supplier aggregation using atomic add.
    
    Processes filtered lineitem rows - each row has a pre-computed
    position (pos) into the supplier array.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load revenue and position
    revenue = tl.load(revenue_ptr + offs, mask=mask, other=0.0)
    pos = tl.load(pos_ptr + offs, mask=mask, other=-1)

    # Only accumulate for valid positions (>= 0)
    valid = (pos >= 0) & mask
    safe_pos = tl.where(valid, pos, 0)
    safe_revenue = tl.where(valid, revenue, 0.0)

    tl.atomic_add(out_ptr + safe_pos, safe_revenue)
