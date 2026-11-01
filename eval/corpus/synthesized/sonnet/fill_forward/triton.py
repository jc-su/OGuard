import torch
import triton
import triton.language as tl


@triton.jit
def _latest(has_a, val_a, has_b, val_b):
    # combine(a, b) = b if b has a value else a
    return has_a | has_b, tl.where(has_b != 0, val_b, val_a)


@triton.jit(do_not_specialize=["n"])
def kernel(values, valid, out, n, BLOCK: tl.constexpr):
    # One program scans the whole column: the kernel has no scratch memory for a
    # cross-program carry.
    # a launch passes a scalar that fits in 32 bits as int32
    n = n.to(tl.int64)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    carry = n * 0
    for start in range(0, n, BLOCK):
        rows = start + lane
        m = rows < n
        byte = tl.load(valid + (rows >> 3), mask=m, other=0).to(tl.int32)
        has = (byte >> (rows & 7).to(tl.int32)) & 1
        val = tl.where(has != 0, tl.load(values + rows, mask=m & (has != 0), other=0), 0)
        scan_has, scan_val = tl.associative_scan((has, val), 0, _latest)
        state = tl.where(scan_has != 0, scan_val, carry)
        tl.store(out + rows, state, mask=m)
        carry = tl.sum(tl.where(lane == BLOCK - 1, state, 0), axis=0)


TYPES = {
    "values": "*i64",
    "valid": "*u8",
    "out": "*i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(values, valid, out, n):
    kernel[(1,)](values, valid, out, n, **CONSTANTS, num_warps=NUM_WARPS)
