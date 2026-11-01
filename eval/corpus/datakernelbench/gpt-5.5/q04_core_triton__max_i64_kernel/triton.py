import triton
import triton.language as tl

TYPES = {
    "in_ptr": "*i64",
    "out_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.load(in_ptr + offs, mask=mask, other=0)
    m = tl.max(vals, axis=0)
    tl.store(out_ptr + pid, m)
