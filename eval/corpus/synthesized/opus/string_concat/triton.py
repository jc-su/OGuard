"""string_concat: out_chars[out_offsets[i] : out_offsets[i + 1]] = string i of a followed by
string i of b.

The work is split by output byte, so a long string does not serialize one lane.
Each program loops over tiles of BLOCK output bytes. For a tile it finds the first
and last output row by a scalar binary search over out_offsets, then the row of
every byte by a vector binary search within that range; byte k of row i comes from
a when k < length of a[i] and from b otherwise.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
PROGRAMS_PER_SM = 8

TYPES = {
    "a_offsets": "*i32",
    "a_chars": "*u8",
    "b_offsets": "*i32",
    "b_chars": "*u8",
    "out_offsets": "*i32",
    "out_chars": "*u8",
    "n": "i64",
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
def kernel(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n, BLOCK: tl.constexpr):
    n = tl.cast(n, tl.int64)
    total = tl.load(out_offsets + n).to(tl.int64)
    start = tl.program_id(0).to(tl.int64) * BLOCK
    step = tl.num_programs(0).to(tl.int64) * BLOCK
    zero = tl.zeros((), tl.int64)
    for t0 in range(start, total, step):
        t1 = tl.minimum(t0 + BLOCK, total)
        row_lo = _upper_bound_scalar(out_offsets, zero, n + 1, t0) - 1
        row_hi = _upper_bound_scalar(out_offsets, zero, n + 1, t1 - 1) - 1
        p = t0 + tl.arange(0, BLOCK)
        active = p < t1

        lo = tl.zeros((BLOCK,), tl.int64) + row_lo
        hi = tl.where(active, row_hi + 1, lo)
        while tl.max(hi - lo, axis=0) > 0:
            live = lo < hi
            mid = lo + ((hi - lo) >> 1)
            v = tl.load(out_offsets + mid, mask=live, other=0).to(tl.int64)
            lo = tl.where(live & (v <= p), mid + 1, lo)
            hi = tl.where(live & (v > p), mid, hi)
        r = lo - 1

        k = p - tl.load(out_offsets + r, mask=active, other=0).to(tl.int64)
        a_begin = tl.load(a_offsets + r, mask=active, other=0).to(tl.int64)
        a_len = tl.load(a_offsets + r + 1, mask=active, other=0).to(tl.int64) - a_begin
        b_begin = tl.load(b_offsets + r, mask=active, other=0).to(tl.int64)
        from_a = k < a_len
        ca = tl.load(a_chars + a_begin + k, mask=active & from_a, other=0)
        cb = tl.load(b_chars + b_begin + (k - a_len), mask=active & ~from_a, other=0)
        tl.store(out_chars + p, tl.where(from_a, ca, cb), mask=active)


def run(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    sms = torch.cuda.get_device_properties(out_offsets.device).multi_processor_count
    grid = (sms * PROGRAMS_PER_SM,)
    kernel[grid](a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
