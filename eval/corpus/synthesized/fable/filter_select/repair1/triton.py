"""filter_select: sel[0:count] = row ids i with quantity[i] < threshold.

Program-aggregated compaction. Each program evaluates the predicate on a tile
of BLOCK rows, reserves the selected rows' slots with one atomicAdd on count,
and writes each selected row id at base + (number of selected rows before it
in the tile), which an exclusive tile scan gives.

The output position is formed in 64-bit from base and compared against 0 and
n before the store, so every store stays inside sel (n slots) whatever count
holds on entry. Under the contract (count zero on entry) both comparisons
always hold, because at most n rows are selected.

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
        pos = base.to(tl.int64) + (tl.cumsum(flags, axis=0) - flags)
        writable = pred & (pos >= 0) & (pos < n)
        tl.store(sel + pos, rows.to(tl.int32), mask=writable)


def run(quantity, threshold, sel, count, n):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](quantity, threshold, sel, count, n, **CONSTANTS, num_warps=NUM_WARPS)
