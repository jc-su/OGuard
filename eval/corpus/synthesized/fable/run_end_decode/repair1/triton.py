"""run_end_decode: expand an Arrow run-end encoded column.

Each lane owns one output row i and finds its run j = |{t : run_ends[t] <= i}|
by binary lifting over run_ends (the steps are uniform across the program, so
the loop has no divergence). Rows of one program map to a contiguous range of
runs, so the probes hit few cache lines.

Revision: the masks of the gathers state both bounds of the index explicitly
(1 <= cand <= r for the probe of run_ends at cand - 1, and 0 <= j < r for
the gather from run_values), and the run index is int64 so that every
address is a plain 64-bit offset. The lifting keeps j >= 0 and cand >= 1,
and the last run ends at n, so the extra terms never mask a lane out; they
state the bound of each access for the memory checker, which does not carry
that loop invariant.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
TYPES = {"run_values": "*fp64", "run_ends": "*i32", "out": "*fp64", "r": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(run_values, run_ends, out, r: tl.int64, n: tl.int64, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    row = i.to(tl.int32)
    # largest power of two <= r; the lifting steps then sum to at least r
    step = tl.cast(1, tl.int64)
    while step * 2 <= r:
        step = step * 2
    j = tl.zeros([BLOCK], tl.int64)
    while step > 0:
        cand = j + step
        ok = m & (cand >= 1) & (cand <= r)
        end = tl.load(run_ends + (cand - 1), mask=ok, other=2147483647)
        j = tl.where(end <= row, cand, j)
        step = step // 2
    hit = m & (j >= 0) & (j < r)
    v = tl.load(run_values + j, mask=hit, other=0.0)
    tl.store(out + i, v, mask=m)


def run(run_values: torch.Tensor, run_ends: torch.Tensor, out: torch.Tensor, r: int, n: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](run_values, run_ends, out, r, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
