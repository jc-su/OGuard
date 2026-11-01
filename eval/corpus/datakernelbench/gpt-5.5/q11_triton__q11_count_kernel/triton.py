import triton
import triton.language as tl

TYPES = {
    "group_values_ptr": "*fp64",
    "total_ptr": "*fp64",
    "fraction_ptr": "*fp64",
    "count_ptr": "*i32",
    "n_groups": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    group_values_ptr,
    total_ptr,
    fraction_ptr,
    count_ptr,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_groups

    threshold = tl.load(total_ptr) * tl.load(fraction_ptr)
    vals = tl.load(group_values_ptr + offs, mask=mask, other=0.0)
    keep = mask & (vals > threshold)

    cnt = tl.sum(tl.where(keep, 1, 0))
    tl.atomic_add(count_ptr, cnt, sem="relaxed")
