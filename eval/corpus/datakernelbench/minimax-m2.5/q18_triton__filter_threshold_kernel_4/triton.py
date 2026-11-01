import triton
import triton.language as tl

TYPES = {
    "sums_ptr": "*fp64",
    "threshold": "fp32",
    "out_ptr": "*i32",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 8


@triton.jit
def kernel(
    sums_ptr, threshold, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    """Apply threshold filter using Triton for reduced Python overhead."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    sums = tl.load(sums_ptr + offs, mask=mask, other=0.0)
    keep = mask & (sums > threshold)
    
    result = tl.where(keep, 1, 0)
    tl.store(out_ptr + offs, result, mask=mask)
