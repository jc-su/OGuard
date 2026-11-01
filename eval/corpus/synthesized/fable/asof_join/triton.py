import triton
import triton.language as tl


@triton.jit
def kernel(left, right, out, n, r, tolerance, BLOCK: tl.constexpr):
    # Each program handles BLOCK left rows: a binary search for the last right
    # row at or before the left value, then the tolerance test.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    x = tl.load(left + offs, mask=m, other=0)
    lo = tl.zeros([BLOCK], tl.int64)
    hi = tl.where(m, r, 0) + tl.zeros([BLOCK], tl.int64)
    while tl.max(hi - lo, axis=0) > 0:
        mid = (lo + hi) >> 1
        live = lo < hi
        v = tl.load(right + mid, mask=live, other=0)
        take = live & (v <= x)
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(live & (v > x), mid, hi)
    j = lo - 1
    rj = tl.load(right + j, mask=m & (j >= 0), other=0)
    ok = m & (j >= 0) & ((x - rj) <= tolerance)
    res = tl.where(ok, j, -1).to(tl.int32)
    tl.store(out + offs, res, mask=m)


TYPES = {"left": "*i64", "right": "*i64", "out": "*i32", "n": "i64", "r": "i64", "tolerance": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(left, right, out, n, r, tolerance):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](left, right, out, n, r, tolerance, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
