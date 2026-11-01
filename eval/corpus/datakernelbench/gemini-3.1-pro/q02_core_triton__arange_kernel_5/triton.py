import triton
import triton.language as tl

TYPES = {
    "out_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 4


@triton.jit
def kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, offs, mask=mask)
