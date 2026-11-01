"""scatter_by_offsets: out[offsets[i]] = i for every i with flags[i] != 0.

offsets is the exclusive prefix sum of flags, so the selected rows land in
increasing row order. One tile of BLOCK rows per program.

The position is widened to 64-bit and compared against 0 and total before
the store, so every store stays inside out (total slots) for any offsets.
Under the contract both comparisons always hold for a flagged row, because
its exclusive prefix sum is below the sum of all flags.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {"flags": "*i8", "offsets": "*i32", "out": "*i32", "n": "i64", "total": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(flags, offsets, out, n: tl.int64, total: tl.int64, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = rows < n
    f = tl.load(flags + rows, mask=in_bounds, other=0)
    selected = in_bounds & (f != 0)
    dst = tl.load(offsets + rows, mask=selected, other=0).to(tl.int64)
    writable = selected & (dst >= 0) & (dst < total)
    tl.store(out + dst, rows.to(tl.int32), mask=writable)


def run(flags, offsets, out, n, total):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](flags, offsets, out, n, total, **CONSTANTS, num_warps=NUM_WARPS)
