import triton
import triton.language as tl

TYPES = {
    "a_ptr": "*i32",
    "b_ptr": "*i32",
    "c_ptr": "*i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    tl.store(a_ptr + offs, z, mask=mask)
    tl.store(b_ptr + offs, z, mask=mask)
    tl.store(c_ptr + offs, z, mask=mask)
