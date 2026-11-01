import triton
import triton.language as tl

TYPES = {
    "unique_packed_ptr": "*i64",
    "counts_ptr": "*i64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    unique_packed_ptr, counts_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    p = tl.load(unique_packed_ptr + offs, mask=mask, other=0)
    g = p >> 32
    
    val = tl.full([BLOCK_SIZE], 1, dtype=tl.int64)
    tl.atomic_add(counts_ptr + g, val, mask=mask)
