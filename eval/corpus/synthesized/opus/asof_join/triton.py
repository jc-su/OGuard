"""asof_join: out[i] = largest j with right[j] <= left[i] and left[i] - right[j] <= tolerance,
or -1.

The largest j with right[j] <= left[i] is upper_bound(right, left[i]) - 1, found by
a vector binary search. Any smaller j has right[j] at most as large, so it is
farther from left[i]: if the largest j is out of tolerance, there is no match.
Values are in [-2^61, 2^61], so left[i] - right[j] does not overflow.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {
    "left": "*i64",
    "right": "*i64",
    "out": "*i32",
    "n": "i64",
    "r": "i64",
    "tolerance": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(left, right, out, n, r, tolerance, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    x = tl.load(left + rows, mask=in_range, other=0)
    lo = tl.zeros((BLOCK,), tl.int64)
    hi = tl.where(in_range, lo + r, lo)
    while tl.max(hi - lo, axis=0) > 0:
        active = lo < hi
        mid = lo + ((hi - lo) >> 1)
        v = tl.load(right + mid, mask=active, other=0)
        lo = tl.where(active & (v <= x), mid + 1, lo)
        hi = tl.where(active & (v > x), mid, hi)
    j = lo - 1
    found = tl.load(right + j, mask=in_range & (j >= 0), other=0)
    hit = (j >= 0) & (x - found <= tolerance)
    tl.store(out + rows, tl.where(hit, j, -1).to(tl.int32), mask=in_range)


def run(left, right, out, n, r, tolerance):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](left, right, out, n, r, tolerance, BLOCK=BLOCK, num_warps=NUM_WARPS)
