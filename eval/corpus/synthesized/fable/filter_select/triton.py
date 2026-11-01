"""filter_select: sel[0:count] = row ids i with quantity[i] < threshold.

Program-aggregated compaction. Each program evaluates the predicate on a tile
of BLOCK rows, reserves the selected rows' slots with one atomicAdd on count,
and writes each selected row id at base + (number of selected rows before it
in the tile), which an inclusive tile scan gives.

count must be zero on entry.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {"quantity": "*fp64", "threshold": "fp64", "sel": "*i32", "count": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(quantity, threshold: tl.float64, sel, count, n: tl.int64, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = rows < n
    q = tl.load(quantity + rows, mask=in_bounds, other=0.0)
    pred = in_bounds & (q < threshold)
    flags = pred.to(tl.int32)
    num = tl.sum(flags, axis=0)
    if num > 0:
        base = tl.atomic_add(count, num, sem="relaxed")
        pos = base + tl.cumsum(flags, axis=0) - 1
        tl.store(sel + pos, rows.to(tl.int32), mask=pred)


def run(quantity, threshold, sel, count, n):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](quantity, threshold, sel, count, n, **CONSTANTS, num_warps=NUM_WARPS)
