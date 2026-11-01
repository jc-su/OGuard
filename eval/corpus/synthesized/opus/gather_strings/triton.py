"""gather_strings: out_chars[out_offsets[i] : out_offsets[i + 1]] = bytes of string idx[i].

The work is split by output byte, so a long string does not serialize one lane.
Each program loops over tiles of BLOCK output bytes. For a tile it finds the first
and last output row by a scalar binary search over out_offsets, then finds the row
of every byte by a vector binary search within that range, and copies the byte
from its source string.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
PROGRAMS_PER_SM = 8

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "idx": "*i32",
    "out_offsets": "*i32",
    "out_chars": "*u8",
    "n": "i64",
    "m": "i64",
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
def _upper_bound_vector(a, lo, hi, x):
    """Per lane: first index k in [lo, hi) with a[k] > x, or hi."""
    while tl.max(hi - lo, axis=0) > 0:
        active = lo < hi
        mid = lo + ((hi - lo) >> 1)
        v = tl.load(a + mid, mask=active, other=0).to(tl.int64)
        lo = tl.where(active & (v <= x), mid + 1, lo)
        hi = tl.where(active & (v > x), mid, hi)
    return lo


@triton.jit
def kernel(offsets, chars, idx, out_offsets, out_chars, n, m, BLOCK: tl.constexpr):
    m = tl.cast(m, tl.int64)
    total = tl.load(out_offsets + m).to(tl.int64)
    start = tl.program_id(0).to(tl.int64) * BLOCK
    step = tl.num_programs(0).to(tl.int64) * BLOCK
    zero = tl.zeros((), tl.int64)
    for t0 in range(start, total, step):
        t1 = tl.minimum(t0 + BLOCK, total)
        row_lo = _upper_bound_scalar(out_offsets, zero, m + 1, t0) - 1
        row_hi = _upper_bound_scalar(out_offsets, zero, m + 1, t1 - 1) - 1
        p = t0 + tl.arange(0, BLOCK)
        active = p < t1
        lo = tl.zeros((BLOCK,), tl.int64) + row_lo
        hi = tl.where(active, row_hi + 1, lo)
        r = _upper_bound_vector(out_offsets, lo, hi, p) - 1
        r_start = tl.load(out_offsets + r, mask=active, other=0).to(tl.int64)
        row = tl.load(idx + r, mask=active, other=0)
        src = tl.load(offsets + row, mask=active, other=0).to(tl.int64) + (p - r_start)
        c = tl.load(chars + src, mask=active, other=0)
        tl.store(out_chars + p, c, mask=active)


def run(offsets, chars, idx, out_offsets, out_chars, n, m):
    sms = torch.cuda.get_device_properties(out_offsets.device).multi_processor_count
    grid = (sms * PROGRAMS_PER_SM,)
    kernel[grid](offsets, chars, idx, out_offsets, out_chars, n, m, BLOCK=BLOCK, num_warps=NUM_WARPS)
