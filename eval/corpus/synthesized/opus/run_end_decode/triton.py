"""Run-end decode of an Arrow run-end encoded float64 column.

Each program decodes the BLOCK rows [lo, lo + BLOCK). It first finds the run
that contains row lo, j0 = #{j : run_ends[j] <= lo}, with a PROBES-ary search
over run_ends (every step samples PROBES run ends and keeps the bracket that
holds lo, so r = 10^8 needs three steps). At most BLOCK runs overlap the tile and
they are the runs j0, j0 + 1, ...; the ends of those runs that fall inside the
tile mark where a new run starts, so a histogram of (run end - lo) over BLOCK
bins, followed by an inclusive scan, gives the run offset from j0 of every row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(run_values, run_ends, out, r, n, BLOCK: tl.constexpr, PROBES: tl.constexpr):
    lo = tl.program_id(0).to(tl.int64) * BLOCK
    # j0 = #{j : run_ends[j] <= lo}. Invariant: j0 lies in [first, first + span].
    first = lo * 0
    span = tl.cast(r, tl.int64)
    probe = tl.arange(0, PROBES).to(tl.int64)
    while span > 0:
        step = (span + PROBES - 1) // PROBES
        pos = first + (probe + 1) * step - 1
        sampled = pos < first + span
        end = tl.load(run_ends + pos, mask=sampled, other=0)
        below = tl.sum((sampled & (end <= lo)).to(tl.int64), axis=0)
        first = first + below * step
        span = tl.minimum(step - 1, span - below * step)
    j0 = first

    k = tl.arange(0, BLOCK)
    run = j0 + k
    has_run = run < r
    rel_end = tl.load(run_ends + run, mask=has_run, other=0) - lo
    starts = tl.histogram(rel_end.to(tl.int32), BLOCK, mask=has_run & (rel_end < BLOCK))
    row = lo + k
    in_range = row < n
    value = tl.load(run_values + (j0 + tl.cumsum(starts, axis=0)), mask=in_range)
    tl.store(out + row, value, mask=in_range)


TYPES = {"run_values": "*fp64", "run_ends": "*i32", "out": "*fp64", "r": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": 2048, "PROBES": 512}
NUM_WARPS = 8


def run(
    run_values: torch.Tensor, run_ends: torch.Tensor, out: torch.Tensor, r: int, n: int
) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](run_values, run_ends, out, r, n, num_warps=NUM_WARPS, **CONSTANTS)
