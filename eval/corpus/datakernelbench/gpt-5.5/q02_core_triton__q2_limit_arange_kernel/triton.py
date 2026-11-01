import triton
import triton.language as tl

TYPES = {
    "out_ptr": "*i64",
}

CONSTANTS = {"n_elements": 100, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, offs.to(tl.int64), mask=mask)
