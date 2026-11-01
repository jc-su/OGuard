import triton
import triton.language as tl

TYPES = {
    "totals_ptr": "*fp64",
    "present_ptr": "*u8",
    "count_ptr": "*i32",
}

CONSTANTS = {"n_sup": 100000, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    totals_ptr,
    present_ptr,
    count_ptr,
    n_sup: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_sup
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    zi = tl.zeros((BLOCK_SIZE,), dtype=tl.uint8)
    tl.store(totals_ptr + offs, zf, mask=mask)
    tl.store(present_ptr + offs, zi, mask=mask)
    tl.store(count_ptr, 0, mask=pid == 0)
