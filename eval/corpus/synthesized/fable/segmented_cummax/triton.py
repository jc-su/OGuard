import triton
import triton.language as tl


@triton.jit
def segmax(f1, v1, f2, v2):
    # Segmented max: a partition start on the right discards the left.
    return f1 | f2, tl.where(f2, v2, tl.maximum(v1, v2))


@triton.jit
def kernel(values, first, out, n, BLOCK: tl.constexpr):
    # Each program scans one tile of BLOCK rows. When the partition of the
    # first tile row starts before the tile, the program first reduces the
    # values from that partition start to the tile start to get the carry.
    # The tile itself is a segmented scan over (start flag, value) pairs; the
    # carry applies to the rows before the first partition start of the tile.
    pid = tl.program_id(0).to(tl.int64)
    t0 = pid * BLOCK
    lane = tl.arange(0, BLOCK).to(tl.int64)
    offs = t0 + lane
    m = offs < n
    v = tl.load(values + offs, mask=m, other=0)
    f = tl.load(first + offs, mask=m, other=0).to(tl.int64) == offs
    sf, sv = tl.associative_scan((f, v), 0, segmax)
    carry = tl.full([BLOCK], -0x7FFFFFFFFFFFFFFF - 1, tl.int64)
    seg = tl.load(first + t0, mask=t0 < n, other=0).to(tl.int64)
    if (t0 < n) & (seg < t0):
        a = seg
        while a < t0:
            idx = a + lane
            lm = idx < t0
            w = tl.load(values + idx, mask=lm, other=-0x7FFFFFFFFFFFFFFF - 1)
            carry = tl.maximum(carry, w)
            a = a + BLOCK
        carry = tl.max(carry, axis=0) + tl.zeros([BLOCK], tl.int64)
    res = tl.where(sf, sv, tl.maximum(sv, carry))
    tl.store(out + offs, res, mask=m)


TYPES = {"values": "*i64", "first": "*i32", "out": "*i64", "n": "i64"}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 8


def run(values, first, out, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, first, out, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
