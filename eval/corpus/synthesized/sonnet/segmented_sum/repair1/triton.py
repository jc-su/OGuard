"""Triton kernel for segmented_sum.

sums[j] = sum(values[offsets[j]:offsets[j+1]]) for j in [0, g). One
program is assigned per segment. Each program walks its segment in
chunks of BLOCK values, reducing each chunk with tl.sum, so segments
of very different sizes are handled without one thread per value.

Repair: start = tl.load(offsets + pid), end = tl.load(offsets + pid
+ 1) and the final tl.store(sums + pid, total) were all unmasked.
This file always launches grid=(g,), so pid < g in practice, but the
checker must prove safety for any grid, not just the one this file
happens to launch, so it cannot assume pid stays below g. All three
accesses are now masked on `pid < g`: when pid >= g, start and end
both read as 0 via `other=0`, the walk loop's `pos < end` condition
is immediately false so the loop body never runs, and the store is
skipped, so nothing is read or written outside offsets/sums.
"""

import triton
import triton.language as tl

TYPES = {
    "offsets": "*i32",
    "values": "*fp64",
    "sums": "*fp64",
    "g": "i64",
}

CONSTANTS = {"BLOCK": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(offsets, values, sums, g, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    pid_mask = pid < g

    start = tl.load(offsets + pid, mask=pid_mask, other=0)
    end = tl.load(offsets + pid + 1, mask=pid_mask, other=0)

    total = tl.zeros((), dtype=tl.float64)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK)
        m = offs < end
        v = tl.load(values + offs, mask=m, other=0.0)
        total += tl.sum(v)
        pos += BLOCK

    tl.store(sums + pid, total, mask=pid_mask)


def run(offsets, values, sums, g):
    """Launch kernel on PyTorch CUDA tensors."""
    if g == 0:
        return
    grid = (g,)
    kernel[grid](offsets, values, sums, g, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
