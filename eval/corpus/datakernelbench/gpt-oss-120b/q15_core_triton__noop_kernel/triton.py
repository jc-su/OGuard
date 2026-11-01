import triton
import triton.language as tl

TYPES = {
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 128}
NUM_WARPS = 4


@triton.jit
def kernel(
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    _ = offs < n_elements  # compute mask, discard result
