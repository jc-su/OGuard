import triton
import triton.language as tl

TYPES = {
    "target_ptr": "*fp64",
    "total_ptr": "*fp64",
    "share_ptr": "*fp64",
    "n_years": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 16}

NUM_WARPS = 4


@triton.jit
def kernel(
    target_ptr,
    total_ptr,
    share_ptr,
    n_years,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_years

    total = tl.load(total_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)
    safe_total = tl.where(total != 0.0, total, 1.0)
    share = target / safe_total

    tl.store(share_ptr + offs, share, mask=mask)
