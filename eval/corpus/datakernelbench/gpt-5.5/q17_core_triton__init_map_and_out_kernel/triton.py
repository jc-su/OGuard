import triton
import triton.language as tl

TYPES = {
    "map_ptr": "*i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    map_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(map_ptr + offs, tl.full((BLOCK_SIZE,), -1, tl.int32), mask=mask)
    tl.store(out_ptr, tl.full((), 0.0, tl.float64), mask=pid == 0)
