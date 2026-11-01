import triton
import triton.language as tl

TYPES = {
    "priority_ptr": "*i32",
    "counts_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    priority_ptr,
    counts_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    priority = tl.load(priority_ptr + offs, mask=mask, other=-1)
    tl.atomic_add(counts_ptr + priority, tl.where(mask, 1, 0))
