import triton
import triton.language as tl

TYPES = {
    "ps_availqty_ptr": "*fp64",
    "sum_qty_ptr": "*fp64",
    "out_mask_ptr": "*i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 2048
}

NUM_WARPS = 4


@triton.jit
def kernel(
    ps_availqty_ptr, sum_qty_ptr,
    out_mask_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Q20 comparison: ps_availqty > 0.5 * sum_qty AND sum_qty > 0
    Fuses the comparison into a single GPU pass for memory-bound scan optimization.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load data with safe defaults for OOB lanes
    avail = tl.load(ps_availqty_ptr + offs, mask=mask, other=0.0)
    sum_q = tl.load(sum_qty_ptr + offs, mask=mask, other=0.0)
    
    # Compare: ps_availqty > 0.5 * sum_qty AND sum_qty > 0
    keep = (avail > 0.5 * sum_q) & (sum_q > 0)
    
    # Store mask (int32 for CUDA compatibility)
    tl.store(out_mask_ptr + offs, keep.to(tl.int32), mask=mask)
