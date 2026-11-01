import triton
import triton.language as tl

TYPES = {
    "counts_ptr": "*i32",
    "hist_ptr": "*i32",
    "counts_n": "i32",
}

CONSTANTS = {"HIST_BINS": 64, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    counts_ptr,
    hist_ptr,
    counts_n,
    HIST_BINS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    tl.store(counts_ptr + offs, z, mask=offs < counts_n)
    tl.store(hist_ptr + offs, z, mask=offs < HIST_BINS)
