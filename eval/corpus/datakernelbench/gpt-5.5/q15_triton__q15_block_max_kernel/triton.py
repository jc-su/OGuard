import triton
import triton.language as tl

TYPES = {
    "totals_ptr": "*fp64",
    "present_ptr": "*u8",
    "block_max_ptr": "*fp64",
}

CONSTANTS = {"n_sup": 100000, "BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    totals_ptr,
    present_ptr,
    block_max_ptr,
    n_sup: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_sup
    present = tl.load(present_ptr + offs, mask=mask, other=0)
    vals = tl.load(totals_ptr + offs, mask=mask, other=-float("inf"))
    vals = tl.where(mask & (present != 0), vals, -float("inf"))
    m = tl.max(vals, axis=0)
    tl.store(block_max_ptr + pid, m)
