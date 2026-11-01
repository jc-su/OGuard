"""join_materialize: for probe row i whose key matches build rows f .. f + c - 1,
out_build[offsets[i] + j] = f + j and out_probe[offsets[i] + j] = i.

The work is split by output position, so a probe row with many matches does not
serialize one lane. Each program loops over tiles of BLOCK output positions. For a
tile it finds the first and last probe row by a scalar binary search over offsets,
then the probe row of every position by a vector binary search within that range,
and f by a vector binary search (lower bound) of probe[i] in the sorted build
column.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
PROGRAMS_PER_SM = 8

TYPES = {
    "build": "*i64",
    "probe": "*i64",
    "offsets": "*i32",
    "out_build": "*i32",
    "out_probe": "*i32",
    "b": "i64",
    "p": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def _upper_bound_scalar(a, lo, hi, x):
    """First index k in [lo, hi) with a[k] > x, or hi."""
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        go = tl.load(a + mid).to(tl.int64) <= x
        lo = tl.where(go, mid + 1, lo)
        hi = tl.where(go, hi, mid)
    return lo


@triton.jit
def kernel(build, probe, offsets, out_build, out_probe, b, p, BLOCK: tl.constexpr):
    p = tl.cast(p, tl.int64)
    total = tl.load(offsets + p).to(tl.int64)
    start = tl.program_id(0).to(tl.int64) * BLOCK
    step = tl.num_programs(0).to(tl.int64) * BLOCK
    zero = tl.zeros((), tl.int64)
    for t0 in range(start, total, step):
        t1 = tl.minimum(t0 + BLOCK, total)
        row_lo = _upper_bound_scalar(offsets, zero, p + 1, t0) - 1
        row_hi = _upper_bound_scalar(offsets, zero, p + 1, t1 - 1) - 1
        q = t0 + tl.arange(0, BLOCK)
        active = q < t1

        # Probe row of each position: upper_bound(offsets, q) - 1 within [row_lo, row_hi].
        lo = tl.zeros((BLOCK,), tl.int64) + row_lo
        hi = tl.where(active, row_hi + 1, lo)
        while tl.max(hi - lo, axis=0) > 0:
            live = lo < hi
            mid = lo + ((hi - lo) >> 1)
            v = tl.load(offsets + mid, mask=live, other=0).to(tl.int64)
            lo = tl.where(live & (v <= q), mid + 1, lo)
            hi = tl.where(live & (v > q), mid, hi)
        r = lo - 1
        j = q - tl.load(offsets + r, mask=active, other=0).to(tl.int64)
        key = tl.load(probe + r, mask=active, other=0)

        # First build row of the key: lower_bound(build, key).
        lo = tl.zeros((BLOCK,), tl.int64)
        hi = tl.where(active, lo + b, lo)
        while tl.max(hi - lo, axis=0) > 0:
            live = lo < hi
            mid = lo + ((hi - lo) >> 1)
            v = tl.load(build + mid, mask=live, other=0)
            lo = tl.where(live & (v < key), mid + 1, lo)
            hi = tl.where(live & (v >= key), mid, hi)

        tl.store(out_build + q, (lo + j).to(tl.int32), mask=active)
        tl.store(out_probe + q, r.to(tl.int32), mask=active)


def run(build, probe, offsets, out_build, out_probe, b, p):
    sms = torch.cuda.get_device_properties(offsets.device).multi_processor_count
    grid = (sms * PROGRAMS_PER_SM,)
    kernel[grid](build, probe, offsets, out_build, out_probe, b, p, BLOCK=BLOCK, num_warps=NUM_WARPS)
