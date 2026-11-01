"""Run-end decode of an Arrow run-end encoded float64 column.

Each program decodes the BLOCK rows [lo, lo + BLOCK). It first finds the run
that contains row lo, j0 = #{j : run_ends[j] <= lo}, with a PROBES-ary search
over run_ends: every step samples PROBES run ends and keeps the bracket that
holds j0, and STEPS = 4 steps narrow any r below 2^31 to one position. At
most BLOCK runs overlap the tile
and they are the runs j0, j0 + 1, ...; the ends of those runs that fall inside
the tile mark where a new run starts, so a histogram of (run end - lo) over
BLOCK bins, followed by an inclusive scan, gives the run offset from j0 of
every row. The end of a run past r - 1 or past the tile goes to bin 0, which
no run end inside the tile uses (they are all above lo), and the scan result
drops those entries again. The histogram takes no mask, because a mask makes
Triton guard each shared-memory atomic with a branch.

Bounds: every load and store has the mask 0 <= index and index < length of
the very index it uses (r for run_ends and run_values, n for out), so its
bound needs no minimum, maximum or loop invariant. The search keeps the last
known position before j0 (or -1) and probes last + i * step, and a histogram
bin is masked with BLOCK - 1.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    run_values,
    run_ends,
    out,
    r,
    n,
    BLOCK: tl.constexpr,
    PROBES: tl.constexpr,
    STEPS: tl.constexpr,
):
    lo = tl.program_id(0).to(tl.int64) * BLOCK
    # j0 = #{j : run_ends[j] <= lo}. Invariant: j0 lies in [last + 1, last + 1 + span].
    # A fixed number of steps keeps the loads free of a loop guard on r.
    last = lo * 0 - 1
    span = tl.cast(r, tl.int64)
    probe = tl.arange(0, PROBES).to(tl.int64)
    for _ in tl.static_range(STEPS):
        # ceil(span / PROBES): at least 1 while span > 0, and 0 once the bracket is found.
        step = (span + PROBES - 1) // PROBES
        pos = last + (probe + 1) * step
        sampled = pos <= last + span
        end = tl.load(run_ends + pos, mask=sampled & (pos >= 0) & (pos < r), other=0)
        below = tl.sum((sampled & (end <= lo)).to(tl.int64), axis=0)
        last = last + below * step
        span = tl.minimum(step - 1, span - below * step)
    j0 = last + 1

    k = tl.arange(0, BLOCK)
    run = j0 + k
    end = tl.load(run_ends + run, mask=(run >= 0) & (run < r), other=0)
    rel_end = end - lo
    inside = (run < r) & (rel_end > 0) & (rel_end < BLOCK)
    bin = tl.where(inside, rel_end, 0).to(tl.int32) & (BLOCK - 1)
    outside = tl.sum((~inside).to(tl.int32), axis=0)
    value_index = j0 + tl.cumsum(tl.histogram(bin, BLOCK), axis=0) - outside
    value = tl.load(
        run_values + value_index, mask=(value_index >= 0) & (value_index < r), other=0.0
    )
    row = lo + k
    tl.store(out + row, value, mask=(row >= 0) & (row < n))


TYPES = {"run_values": "*fp64", "run_ends": "*i32", "out": "*fp64", "r": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": 2048, "PROBES": 512, "STEPS": 4}  # 512^4 >= 2^31
NUM_WARPS = 8


def run(
    run_values: torch.Tensor, run_ends: torch.Tensor, out: torch.Tensor, r: int, n: int
) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](run_values, run_ends, out, r, n, num_warps=NUM_WARPS, **CONSTANTS)
