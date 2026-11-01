"""group_argmax: best[k] = smallest row of group k whose value is the largest of the
group; stays -1 for a group without rows.

best[k] holds a row; candidates are ordered by (value desc, row asc) and the row is
replaced by compare-and-swap, comparing against value[best[k]], so best[k] only
improves. Each lane reads best[k] first and drops out when its candidate does not
win; a stale read only shows an older (worse) row. When all rows of a program
belong to one group (few groups, or sorted groups), the program first reduces them
to one candidate, so it issues one update instead of BLOCK. tl.atomic_cas has no
mask, so lanes that are done issue CAS(slot, cur, cur), which leaves the slot
unchanged.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {
    "group": "*i32",
    "value": "*i64",
    "best": "*i32",
    "n": "i64",
    "g": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def _pick(v1, r1, v2, r2):
    first = (v1 > v2) | ((v1 == v2) & (r1 < r2))
    return tl.where(first, v1, v2), tl.where(first, r1, r2)


@triton.jit
def kernel(group, value, best, n, g, BLOCK: tl.constexpr):
    INT64_MIN: tl.constexpr = -9223372036854775808
    INT64_MAX: tl.constexpr = 9223372036854775807
    lane = tl.arange(0, BLOCK)
    rows = tl.program_id(0).to(tl.int64) * BLOCK + lane
    in_range = rows < n
    k = tl.load(group + rows, mask=in_range, other=0)
    v = tl.load(value + rows, mask=in_range, other=INT64_MIN)
    r = tl.where(in_range, rows, INT64_MAX)

    # One group in the whole program: reduce to one candidate in lane 0.
    uniform = tl.min(tl.where(in_range, k, 2147483647), axis=0) == tl.max(tl.where(in_range, k, -1), axis=0)
    bv, br = tl.reduce((v, r), 0, _pick)
    lead = uniform & (lane == 0)
    v = tl.where(lead, bv, v)
    r = tl.where(lead, br, r)
    active = tl.where(uniform, lane == 0, in_range)

    r32 = r.to(tl.int32)
    cur = tl.load(best + k, mask=active, other=0)
    while tl.max(active.to(tl.int32), axis=0) > 0:
        cv = tl.load(value + tl.maximum(cur, 0), mask=active & (cur >= 0), other=0)
        loses = (cur >= 0) & ((cv > v) | ((cv == v) & (cur <= r)))
        active = active & ~loses
        old = tl.atomic_cas(best + k, cur, tl.where(active, r32, cur))
        done = active & (old == cur)
        cur = tl.where(active, old, cur)
        active = active & ~done


def run(group, value, best, n, g):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](group, value, best, n, g, BLOCK=BLOCK, num_warps=NUM_WARPS)
