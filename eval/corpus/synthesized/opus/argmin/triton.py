"""argmin: row[0] = smallest i such that values[i] is the minimum of values; stays -1
for n = 0.

Each program keeps a per-lane best (value, row) over its tiles, reduces it to one
pair ordered by value and then by row, and publishes it with a compare-and-swap
loop on row[0]. The loop compares against values[row[0]], so the published row
only ever improves; a failed CAS retries with the value it returned.
"""

import torch
import triton
import triton.language as tl

BLOCK = 2048
NUM_WARPS = 8
PROGRAMS_PER_SM = 8

TYPES = {
    "values": "*i64",
    "row": "*i32",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def _pick_better(v1, r1, v2, r2):
    first = (v1 < v2) | ((v1 == v2) & (r1 < r2))
    return tl.where(first, v1, v2), tl.where(first, r1, r2)


@triton.jit
def kernel(values, row, n, BLOCK: tl.constexpr):
    INT64_MAX: tl.constexpr = 9223372036854775807
    lane = tl.arange(0, BLOCK)
    best_v = tl.full((BLOCK,), INT64_MAX, tl.int64)
    best_r = tl.full((BLOCK,), INT64_MAX, tl.int64)  # no row
    start = tl.program_id(0).to(tl.int64) * BLOCK
    step = tl.num_programs(0).to(tl.int64) * BLOCK
    for base in range(start, n, step):
        rows = base + lane
        in_range = rows < n
        v = tl.load(values + rows, mask=in_range, other=0)
        # Rows of one lane increase, so a strict comparison keeps the first of equal values.
        take = in_range & ((v < best_v) | (best_r == INT64_MAX))
        best_v = tl.where(take, v, best_v)
        best_r = tl.where(take, rows, best_r)
    bv, br = tl.reduce((best_v, best_r), 0, _pick_better)

    # br == INT64_MAX when the program had no rows; then the loop does not run.
    # Scalars must be the same in every thread of the program: a plain load of row[0]
    # could differ between threads while other programs update it, but the result of
    # a scalar atomic is broadcast. So the loop starts from the guess cur = -1 and
    # takes row[0] from the first CAS.
    r32 = br.to(tl.int32)
    cur = tl.full((), -1, tl.int32)
    cv = tl.zeros((), tl.int64)
    while (br < n) & ((cur < 0) | (bv < cv) | ((bv == cv) & (br < cur))):
        old = tl.atomic_cas(row, cur, r32)
        cur = tl.where(old == cur, r32, old)
        cv = tl.load(values + tl.maximum(cur, 0).to(tl.int64))


def run(values, row, n):
    sms = torch.cuda.get_device_properties(values.device).multi_processor_count
    grid = (max(1, min(triton.cdiv(n, BLOCK), sms * PROGRAMS_PER_SM)),)
    kernel[grid](values, row, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
