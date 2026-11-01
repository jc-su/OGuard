import torch
import triton
import triton.language as tl


@triton.jit
def kernel(left, right, out, n, r, tolerance, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    inside = offs < n
    x = tl.load(left + offs, mask=inside, other=0)
    lo = tl.zeros([BLOCK], tl.int64)
    hi = tl.zeros([BLOCK], tl.int64) + r
    # The search interval of r rows halves at least every step, so bit_length(r)
    # steps end every search.
    steps = r * 0
    size = r
    while size > 0:
        size = size >> 1
        steps += 1
    for _ in range(steps):
        active = lo < hi
        mid = lo + (hi - lo) // 2
        probe = tl.load(right + mid, mask=active, other=0)
        go_right = active & (probe <= x)
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(active & (probe > x), mid, hi)
    # The largest row with right[j] <= x is the closest to x, so if it is
    # outside the tolerance every smaller row is too.
    j = lo - 1
    nearest = tl.load(right + j, mask=inside & (j >= 0), other=0)
    found = (j >= 0) & ((x - nearest) <= tolerance)
    result = tl.where(found, j, -1)
    tl.store(out + offs, result.to(tl.int32), mask=inside)


TYPES = {
    "left": "*i64",
    "right": "*i64",
    "out": "*i32",
    "n": "i64",
    "r": "i64",
    "tolerance": "i64",
}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(left, right, out, n, r, tolerance):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](left, right, out, n, r, tolerance, **CONSTANTS, num_warps=NUM_WARPS)
