import triton
import triton.language as tl


@triton.jit
def kernel(group, value, best, n, g, BLOCK: tl.constexpr):
    # Each program handles BLOCK rows, one lane per row, with a
    # compare-and-swap loop on best[group] that keeps the row with the larger
    # value, or the smaller row on a tie. A lane whose published row is at
    # least as good as its own is done; lanes that are done make the
    # compare-and-swap a no-op by comparing against and writing the value
    # they already saw.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    k = tl.load(group + offs, mask=m, other=0).to(tl.int64)
    v = tl.load(value + offs, mask=m, other=0)
    row = offs.to(tl.int32)
    cur = tl.load(best + k, mask=m, other=0)
    cv = tl.load(value + cur.to(tl.int64), mask=m & (cur >= 0), other=0)
    active = m & ((cur < 0) | (cv < v) | ((cv == v) & (cur > row)))
    while tl.sum(active.to(tl.int32), axis=0) > 0:
        old = tl.atomic_cas(best + k, cur, tl.where(active, row, cur))
        active = active & (old != cur)
        cur = old
        cv = tl.load(value + cur.to(tl.int64), mask=active & (cur >= 0), other=0)
        active = active & ((cur < 0) | (cv < v) | ((cv == v) & (cur > row)))


TYPES = {"group": "*i32", "value": "*i64", "best": "*i32", "n": "i64", "g": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(group, value, best, n, g):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](group, value, best, n, g, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
