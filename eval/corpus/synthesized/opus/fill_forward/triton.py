"""fill_forward: out[i] = values[j] for the largest valid row j <= i, or 0 when rows
0..i are all null.

A run of nulls can span the whole column, and Triton has no grid-wide barrier and
the kernel has no scratch memory to pass carries between programs, so one program
scans the column in tiles of BLOCK rows. Each tile is an associative scan of
(valid flag, value) pairs that keeps the value of the last valid row; rows before
the first valid row of the tile take the carry, the result of the previous tile's
last row (initially 0).
"""

import triton
import triton.language as tl

BLOCK = 4096
NUM_WARPS = 16

TYPES = {
    "values": "*i64",
    "valid": "*u8",
    "out": "*i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def _last_valid(f1, v1, f2, v2):
    return f1 | f2, tl.where(f2 != 0, v2, v1)


@triton.jit
def kernel(values, valid, out, n, BLOCK: tl.constexpr):
    n = tl.cast(n, tl.int64)
    lane = tl.arange(0, BLOCK)
    carry = tl.zeros((), tl.int64)
    for t0 in range(0, n, BLOCK):
        rows = t0 + lane
        in_range = rows < n
        vbyte = tl.load(valid + (rows >> 3), mask=in_range, other=0).to(tl.int32)
        f = in_range & (((vbyte >> (rows & 7).to(tl.int32)) & 1) != 0)
        v = tl.load(values + rows, mask=f, other=0)
        fs, vs = tl.associative_scan((f.to(tl.int32), v), 0, _last_valid)
        result = tl.where(fs != 0, vs, carry)
        tl.store(out + rows, result, mask=in_range)
        carry = tl.sum(tl.where(lane == BLOCK - 1, result, 0), axis=0)


def run(values, valid, out, n):
    kernel[(1,)](values, valid, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
