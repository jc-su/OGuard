import torch
import triton
import triton.language as tl


# The state of a row is the pair (partition started, running max), combined by
# (f1, v1) . (f2, v2) = (f1 | f2, f2 ? v2 : max(v1, v2)), which is associative.
@triton.jit
def combine(f1, v1, f2, v2):
    return f1 | f2, tl.where(f2 != 0, v2, tl.maximum(v1, v2))


# One program walks the column tile by tile; the running max at the end of a
# tile is the carry into the next one. Work is linear for any partition length.
@triton.jit
def kernel(values, first, out, n, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK).to(tl.int64)
    carry = tl.cast(-9223372036854775807 - 1, tl.int64)
    for start in range(0, n, BLOCK):
        offs = start + lane
        inside = offs < n
        x = tl.load(values + offs, mask=inside, other=0)
        f = (tl.load(first + offs, mask=inside, other=0).to(tl.int64) == offs).to(tl.int32)
        started, run = tl.associative_scan((f, x), axis=0, combine_fn=combine)
        result = tl.where(started != 0, run, tl.maximum(run, carry))
        tl.store(out + offs, result, mask=inside)
        last = tl.minimum(start + BLOCK, n) - 1
        carry = tl.max(tl.where(offs == last, result, -9223372036854775807 - 1), axis=0)


TYPES = {
    "values": "*i64",
    "first": "*i32",
    "out": "*i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(values, first, out, n):
    kernel[(1,)](values, first, out, n, **CONSTANTS, num_warps=NUM_WARPS)
