"""Run-end decode of an Arrow run-end encoded float64 column.

Each program decodes the BLOCK rows [lo, lo + BLOCK). It first finds the run
that contains row lo, j0 = #{j : run_ends[j] <= lo}, with a PROBES-ary search
over run_ends (every step samples PROBES run ends and keeps the bracket that
holds lo, so r = 10^8 needs three steps). At most BLOCK runs overlap the tile and
they are the runs j0, j0 + 1, ...; the ends of those runs that fall inside the
tile mark where a new run starts, so a histogram of (run end - lo) over BLOCK
bins, followed by an inclusive scan, gives the run offset from j0 of every row.

Bounds: every load and store has an index that is at least 0 by construction
and a mask that is one comparison of that index with the length of its array.
The search keeps first >= 0 and step >= 1 with a maximum each (neither changes
a value), so a probe is at least 0, and it is masked with probe < r. A value
index is raised to 0 with a maximum (it is not negative for input that meets
the contract) and masked with index < r; a run index is j0 + k with j0 >= 0
and the mask run < r; a row is lo + k with the mask row < n.
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
        step = tl.maximum((span + PROBES - 1) // PROBES, 1)
        pos = first + (probe + 1) * step - 1
        sampled = pos < first + span
        end = tl.load(run_ends + pos, mask=pos < r, other=0)
        below = tl.sum((sampled & (end <= lo)).to(tl.int64), axis=0)
        first = tl.maximum(first + below * step, 0)
        span = tl.minimum(step - 1, span - below * step)
    j0 = first

    k = tl.arange(0, BLOCK)
    run = j0 + k
    rel_end = tl.load(run_ends + run, mask=run < r, other=0) - lo
    # Runs past r - 1 and ends past the tile fall outside the bins.
    rel_end = tl.where(run < r, rel_end, BLOCK).to(tl.int32)
    starts = tl.histogram(rel_end, BLOCK, mask=rel_end < BLOCK)
    value_index = tl.maximum(j0 + tl.cumsum(starts, axis=0), 0)
    value = tl.load(run_values + value_index, mask=value_index < r, other=0.0)
    row = lo + k
    tl.store(out + row, value, mask=row < n)


TYPES = {"run_values": "*fp64", "run_ends": "*i32", "out": "*fp64", "r": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": 2048, "PROBES": 512}
NUM_WARPS = 8


def run(
    run_values: torch.Tensor, run_ends: torch.Tensor, out: torch.Tensor, r: int, n: int
) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](run_values, run_ends, out, r, n, num_warps=NUM_WARPS, **CONSTANTS)
