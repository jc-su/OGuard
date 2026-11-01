import triton
import triton.language as tl


@triton.jit
def kernel(ts, out, n, unit, BLOCK: tl.constexpr):
    # Each program handles BLOCK rows. The remainder is taken on a
    # nonnegative magnitude, so a timestamp before 1970 rounds toward
    # negative infinity.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    t = tl.load(ts + offs, mask=m, other=0)
    neg = t < 0
    mag = tl.where(neg, 0 - t, t)
    r = mag % unit
    res = tl.where(neg, tl.where(r == 0, t, t - (unit - r)), t - r)
    tl.store(out + offs, res, mask=m)


TYPES = {"ts": "*i64", "out": "*i64", "n": "i64", "unit": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(ts, out, n, unit):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](ts, out, n, unit, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
