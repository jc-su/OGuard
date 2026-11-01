import triton
import triton.language as tl

TYPES = {
    "bal_ptr": "*fp64",
    "mask_ptr": "*u8",
    "out_sum": "*fp64",
    "out_cnt": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    bal_ptr,            # *float64   balance column
    mask_ptr,           # *uint8     1 where row participates in avg
    out_sum,            # *float64   scalar sum (global)
    out_cnt,            # *float64   scalar count (global)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load mask (0/1) as uint8
    m = tl.load(mask_ptr + offs, mask=mask, other=0)
    keep = mask & (m != 0)

    # load balance where keep, else 0.0
    bal = tl.load(bal_ptr + offs, mask=keep, other=0.0)

    # atomic reductions
    tl.atomic_add(out_sum, tl.sum(bal))
    tl.atomic_add(out_cnt, tl.sum(tl.where(keep, 1.0, 0.0)))
