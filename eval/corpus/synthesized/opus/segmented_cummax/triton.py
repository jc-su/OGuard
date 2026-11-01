"""segmented_cummax: out[i] = max(values[first[i] .. i]), a running maximum that restarts
at every partition start (first[i] == i).

A partition can span the whole column, and Triton has no grid-wide barrier and the
kernel has no scratch memory to pass carries between programs, so one program
scans the column in tiles of BLOCK rows. Each tile is a segmented max scan
(associative scan of (start flag, value) pairs); rows before the first partition
start of the tile are combined with the carry, the result of the previous tile's
last row.
"""

import triton
import triton.language as tl

BLOCK = 4096
NUM_WARPS = 16

TYPES = {
    "values": "*i64",
    "first": "*i32",
    "out": "*i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def _segmented_max(f1, v1, f2, v2):
    return f1 | f2, tl.where(f2 != 0, v2, tl.maximum(v1, v2))


@triton.jit
def kernel(values, first, out, n, BLOCK: tl.constexpr):
    NEG_INF: tl.constexpr = -9223372036854775808
    n = tl.cast(n, tl.int64)
    lane = tl.arange(0, BLOCK)
    carry = tl.full((), NEG_INF, tl.int64)
    for t0 in range(0, n, BLOCK):
        rows = t0 + lane
        in_range = rows < n
        v = tl.load(values + rows, mask=in_range, other=NEG_INF)
        start = tl.load(first + rows, mask=in_range, other=0).to(tl.int64) == rows
        f = (start | ~in_range).to(tl.int32)
        fs, vs = tl.associative_scan((f, v), 0, _segmented_max)
        result = tl.where(fs != 0, vs, tl.maximum(vs, carry))
        tl.store(out + rows, result, mask=in_range)
        carry = tl.max(tl.where(lane == BLOCK - 1, result, NEG_INF), axis=0)


def run(values, first, out, n):
    kernel[(1,)](values, first, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
