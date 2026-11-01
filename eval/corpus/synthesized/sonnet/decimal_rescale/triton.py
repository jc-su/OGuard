import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n", "shift"])
def kernel(values, out, overflow, n, shift, BLOCK: tl.constexpr):
    # a launch passes a scalar that fits in 32 bits as int32
    n = n.to(tl.int64)
    shift = shift.to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    v = tl.load(values + offs, mask=m, other=0)

    nonneg = shift >= 0
    k = tl.where(nonneg, shift, -shift)
    pw = shift * 0 + 1
    for _ in range(0, k):
        pw = pw * 10

    # shift >= 0: multiply, flag a product outside int64
    lim = 9223372036854775807 // pw
    ovf = (pw != 1) & ((v > lim) | (v < -lim))
    scaled = tl.where(ovf, 0, v * pw)

    # shift < 0: floor quotient and remainder in [0, pw), then round half away from zero
    q = v // pw
    r = v - q * pw
    adj = r < 0
    q = tl.where(adj, q - 1, q)
    r = tl.where(adj, r + pw, r)
    twice = 2 * r
    up = tl.where(v >= 0, twice >= pw, twice > pw)
    rounded = tl.where(up, q + 1, q)

    if nonneg:
        tl.store(out + offs, scaled, mask=m)
        tl.store(overflow + offs, ovf.to(tl.int8), mask=m)
    else:
        tl.store(out + offs, rounded, mask=m)
        tl.store(overflow + offs, tl.zeros_like(v).to(tl.int8), mask=m)


TYPES = {
    "values": "*i64",
    "out": "*i64",
    "overflow": "*i8",
    "n": "i64",
    "shift": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(values, out, overflow, n, shift):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, out, overflow, n, shift, **CONSTANTS, num_warps=NUM_WARPS)
