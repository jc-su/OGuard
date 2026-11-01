import triton
import triton.language as tl


@triton.jit
def kernel(values, out, overflow, n, shift, BLOCK: tl.constexpr):
    # Each program handles BLOCK rows. The arithmetic runs on the unsigned
    # magnitude so that -2^63 and the overflow test need no special cases.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    v = tl.load(values + offs, mask=m, other=0)
    neg = v < 0
    mag = tl.where(neg, 0 - v, v).to(tl.uint64, bitcast=True)
    one = tl.full([BLOCK], 1, tl.uint64)
    if shift >= 0:
        p = one
        for _ in range(shift):
            p = p * 10
        limit = tl.where(neg, tl.full([BLOCK], 0x8000000000000000, tl.uint64),
                         tl.full([BLOCK], 0x7FFFFFFFFFFFFFFF, tl.uint64))
        ovf = mag > limit // p
        r = mag * p
        r = tl.where(neg, 0 - r, r)
        res = tl.where(ovf, 0, r.to(tl.int64, bitcast=True))
    else:
        q = one
        for _ in range(0 - shift):
            q = q * 10
        qt = mag // q
        rem = mag - qt * q
        qt = qt + (rem * 2 >= q).to(tl.uint64)
        qt = tl.where(neg, 0 - qt, qt)
        res = qt.to(tl.int64, bitcast=True)
        ovf = tl.zeros([BLOCK], tl.int1)
    tl.store(out + offs, res, mask=m)
    tl.store(overflow + offs, ovf.to(tl.int8), mask=m)


TYPES = {"values": "*i64", "out": "*i64", "overflow": "*i8", "n": "i64", "shift": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(values, out, overflow, n, shift):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, out, overflow, n, shift, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
