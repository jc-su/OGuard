import triton
import triton.language as tl

TYPES = {
    "block_max_ptr": "*fp64",
    "max_ptr": "*fp64",
}

CONSTANTS = {"n_blocks": 98, "BLOCK_SIZE": 128}

NUM_WARPS = 4


@triton.jit
def kernel(
    block_max_ptr,
    max_ptr,
    n_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_blocks
    vals = tl.load(block_max_ptr + offs, mask=mask, other=-float("inf"))
    m = tl.max(vals, axis=0)
    tl.store(max_ptr, m)
