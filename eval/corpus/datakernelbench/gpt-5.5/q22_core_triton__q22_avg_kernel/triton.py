import triton
import triton.language as tl

TYPES = {
    "acctbal_ptr": "*fp64",
    "in_mask_ptr": "*i1",
    "stats_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    acctbal_ptr,
    in_mask_ptr,
    stats_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    in_set = tl.load(in_mask_ptr + offs, mask=mask, other=0) != 0
    keep = mask & in_set & (bal > 0.0)

    part_sum = tl.sum(tl.where(keep, bal, 0.0))
    part_cnt = tl.sum(keep.to(tl.float64))

    tl.atomic_add(stats_ptr + 0, part_sum, sem="relaxed")
    tl.atomic_add(stats_ptr + 1, part_cnt, sem="relaxed")
