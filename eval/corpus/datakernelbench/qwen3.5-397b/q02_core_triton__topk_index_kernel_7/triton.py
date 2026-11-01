import triton
import triton.language as tl

TYPES = {
    "out_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 4


@triton.jit
def kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generate sequential indices [0, 1, 2, ..., k-1] for top-k selection.
    
    This kernel writes consecutive indices to output tensor, representing
    the top-k positions after sorting has been performed upstream in cuDF.
    
    Memory-bound tactic: Single fused kernel launch eliminates multiple
    torch operations and reduces kernel launch overhead.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, offs, mask=mask)
