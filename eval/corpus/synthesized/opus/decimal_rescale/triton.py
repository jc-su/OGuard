"""decimal_rescale: change the scale of decimal64 values by 10^shift.

shift >= 0: out = values * 10^shift, or out = 0 and overflow = 1 when the product
leaves the int64 range. For p = 10^shift > 1, p does not divide 2^63, so |v| * p
fits exactly when |v| <= INT64_MAX / p, for either sign.
shift < 0: out = values / 10^-shift rounded to nearest, ties away from zero.
Triton integer division truncates, so q = v // d and rem = v % d has the sign of
v; the result moves one away from zero when 2 * |rem| >= d. |rem| < d <= 10^18,
so 2 * |rem| does not overflow.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {
    "values": "*i64",
    "out": "*i64",
    "overflow": "*i8",
    "n": "i64",
    "shift": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(values, out, overflow, n, shift, BLOCK: tl.constexpr):
    shift = tl.cast(shift, tl.int64)
    k = tl.where(shift >= 0, shift, -shift)
    p = tl.full((), 1, tl.int64)
    for _ in range(k):
        p = p * 10
    limit = 9223372036854775807 // p

    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    v = tl.load(values + rows, mask=in_range, other=0)

    fits = (shift == 0) | ((v >= -limit) & (v <= limit))
    up = tl.where(fits, v * p, 0)
    q = v // p
    rem = v % p
    mag = tl.where(rem < 0, -rem, rem)
    down = q + tl.where(2 * mag >= p, tl.where(v < 0, -1, 1), 0)

    result = tl.where(shift >= 0, up, down)
    flag = (shift >= 0) & ~fits
    tl.store(out + rows, result, mask=in_range)
    tl.store(overflow + rows, flag.to(tl.int8), mask=in_range)


def run(values, out, overflow, n, shift):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](values, out, overflow, n, shift, BLOCK=BLOCK, num_warps=NUM_WARPS)
