import triton
import triton.language as tl


@triton.jit
def take_right(f1, v1, f2, v2):
    # A valid row on the right replaces everything on the left.
    return f1 | f2, tl.where(f2, v2, v1)


@triton.jit
def kernel(values, valid, out, n, BLOCK: tl.constexpr):
    # Each program fills one tile of BLOCK rows. The carry into the tile is
    # the last valid row before it, found by scanning the validity bitmap
    # backward BLOCK bytes at a time; a null run of any length costs one pass
    # over its bitmap bytes per program. The tile itself is a scan with the
    # operator "take the right side when it is valid"; rows before the first
    # valid row of the tile take the carry, or 0 when there is none.
    pid = tl.program_id(0).to(tl.int64)
    t0 = pid * BLOCK
    lane = tl.arange(0, BLOCK).to(tl.int64)
    offs = t0 + lane
    m = offs < n
    byte = tl.load(valid + (offs >> 3), mask=m, other=0).to(tl.int32)
    f = m & (((byte >> (offs & 7).to(tl.int32)) & 1) != 0)
    v = tl.load(values + offs, mask=m, other=0)
    sf, sv = tl.associative_scan((f, v), 0, take_right)
    carry_row = tl.zeros((), tl.int64) - 1
    if (t0 > 0) & (t0 < n):
        top = (t0 - 1) >> 3
        q_hi = top
        while (q_hi >= 0) & (carry_row < 0):
            q = q_hi - lane
            qm = q >= 0
            b = tl.load(valid + q, mask=qm, other=0).to(tl.int32)
            keep = (1 << tl.minimum(tl.maximum(t0 - q * 8, 0), 8).to(tl.int32)) - 1
            b = tl.where(q == top, b & keep, b)
            hb = tl.zeros([BLOCK], tl.int64) - 1
            for bit in tl.static_range(8):
                hb = tl.where(((b >> bit) & 1) != 0, bit, hb)
            carry_row = tl.max(tl.where(qm & (b != 0), q * 8 + hb, -1), axis=0)
            q_hi = q_hi - BLOCK
    carry = tl.load(values + carry_row, mask=carry_row >= 0, other=0)
    res = tl.where(sf, sv, carry)
    tl.store(out + offs, res, mask=m)


TYPES = {"values": "*i64", "valid": "*u8", "out": "*i64", "n": "i64"}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 8


def run(values, valid, out, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, valid, out, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
