"""string_to_int: parse string i as an optional sign followed by one or more ASCII
digits (any number of leading zeros) into int64; out[i] = value and valid bit i = 1
when it parses and fits, else out[i] = 0 and the bit stays 0.

Each lane parses one string; the program steps through byte positions while some
lane is still parsing. The magnitude accumulates in uint64 and is checked before
each step against the limit 2^63 - 1 (or 2^63 for a minus sign):
mag * 10 + d <= limit exactly when mag <= (limit - d) // 10. A lane stops at its
first invalid byte. The program covers BLOCK rows starting at a multiple of BLOCK (a
multiple of 8) and writes their BLOCK // 8 bitmap bytes, so each byte has one
writer and the bits at positions n and above are 0.
"""

import triton
import triton.language as tl

BLOCK = 256
NUM_WARPS = 4

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "out": "*i64",
    "valid": "*u8",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(offsets, chars, out, valid, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    p = tl.load(offsets + rows, mask=in_range, other=0).to(tl.int64)
    end = tl.load(offsets + rows + 1, mask=in_range, other=0).to(tl.int64)
    first = tl.load(chars + p, mask=p < end, other=0)
    sign = (first == 43) | (first == 45)  # '+' or '-'
    neg = first == 45
    p = p + sign.to(tl.int64)
    ok = p < end
    limit = tl.full((BLOCK,), 9223372036854775807, tl.uint64) + neg.to(tl.uint64)
    mag = tl.zeros((BLOCK,), tl.uint64)
    live = ok & (p < end)
    while tl.max(live.to(tl.int32), axis=0) > 0:
        c = tl.load(chars + p, mask=live, other=48).to(tl.int32) - 48
        digit = (c >= 0) & (c <= 9)
        d = tl.where(digit, c, 0).to(tl.uint64)
        fits = mag <= (limit - d) // 10
        step = live & digit & fits
        mag = tl.where(step, mag * 10 + d, mag)
        ok = ok & (~live | step)
        p = p + 1
        live = ok & (p < end)
    value = tl.where(neg, 0 - mag, mag).to(tl.int64, bitcast=True)
    tl.store(out + rows, tl.where(ok, value, 0), mask=in_range)

    bits = tl.reshape(ok.to(tl.int32), (BLOCK // 8, 8)) << tl.arange(0, 8)[None, :]
    byte_val = tl.sum(bits, axis=1)
    out_idx = pid * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
    tl.store(valid + out_idx, byte_val.to(tl.uint8), mask=out_idx < ((n + 7) >> 3))


def run(offsets, chars, out, valid, n):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](offsets, chars, out, valid, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
