import triton
import triton.language as tl


@triton.jit
def kernel(values, bounds, out, n, b, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(values + offs, mask=mask, other=0.0)
    # Binary search for the first bound greater than v: bounds[lo - 1] <= v < bounds[hi].
    lo = tl.zeros([BLOCK], dtype=tl.int64)
    hi = tl.where(mask, b, 0).to(tl.int64)
    span = tl.max(hi - lo, axis=0)
    while span > 0:
        open_ = lo < hi
        mid = lo + (hi - lo) // 2
        bound = tl.load(bounds + mid, mask=open_, other=0.0)
        lo = tl.where(open_ & (bound <= v), mid + 1, lo)
        hi = tl.where(open_ & (bound > v), mid, hi)
        span = tl.max(hi - lo, axis=0)
    tl.store(out + offs, lo.to(tl.int32), mask=mask)


TYPES = {"values": "*fp64", "bounds": "*fp64", "out": "*i32", "n": "i64", "b": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(values, bounds, out, n, b):
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, bounds, out, n, b, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
