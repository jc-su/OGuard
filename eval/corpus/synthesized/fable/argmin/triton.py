import triton
import triton.language as tl


@triton.jit
def kernel(values, row, n, BLOCK: tl.constexpr):
    # Each program reduces BLOCK rows to its (minimum, smallest row) pair and
    # merges the pair into row[0] with a compare-and-swap loop that keeps the
    # pair with the smaller value, or the smaller row on a tie.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = offs < n
    v = tl.load(values + offs, mask=m, other=0x7FFFFFFFFFFFFFFF)
    bv = tl.min(v, axis=0)
    br = tl.min(tl.where(m & (v == bv), offs, 0x7FFFFFFFFFFFFFFF), axis=0)
    if br < n:
        brow = br.to(tl.int32)
        cur = tl.atomic_add(row, 0, sem="acquire")
        cv = tl.load(values + cur.to(tl.int64), mask=cur >= 0, other=0)
        replace = (cur < 0) | (cv > bv) | ((cv == bv) & (cur > brow))
        while replace:
            old = tl.atomic_cas(row, cur, brow)
            done = old == cur
            cur = old
            cv = tl.load(values + cur.to(tl.int64), mask=cur >= 0, other=0)
            replace = (done == 0) & ((cur < 0) | (cv > bv) | ((cv == bv) & (cur > brow)))


TYPES = {"values": "*i64", "row": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 4096}
NUM_WARPS = 8


def run(values, row, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, row, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
