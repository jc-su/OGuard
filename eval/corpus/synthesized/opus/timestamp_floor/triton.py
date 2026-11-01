"""timestamp_floor: out[i] = largest multiple of unit that is <= ts[i] (rounds toward
negative infinity, also before 1970).

Triton integer division truncates toward zero, so the quotient is lowered by one
when the remainder is negative. |ts| <= 2^62 and unit <= 2^40, so q * unit stays in
range.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {
    "ts": "*i64",
    "out": "*i64",
    "n": "i64",
    "unit": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(ts, out, n, unit, BLOCK: tl.constexpr):
    unit = tl.cast(unit, tl.int64)
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    t = tl.load(ts + rows, mask=in_range, other=0)
    q = t // unit
    q = tl.where(t - q * unit < 0, q - 1, q)
    tl.store(out + rows, q * unit, mask=in_range)


def run(ts, out, n, unit):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](ts, out, n, unit, BLOCK=BLOCK, num_warps=NUM_WARPS)
