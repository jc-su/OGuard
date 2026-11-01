import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, bounds, out, n, b, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(values + i, mask=m, other=0.0)
    lo = tl.zeros((BLOCK,), dtype=tl.int64)
    hi = tl.zeros((BLOCK,), dtype=tl.int64) + b
    # binary search for the number of bounds <= v
    while tl.max(hi - lo, axis=0) > 0:
        active = lo < hi
        mid = (lo + hi) >> 1
        x = tl.load(bounds + mid, mask=active, other=0.0)
        le = x <= v
        lo = tl.where(active & le, mid + 1, lo)
        hi = tl.where(active & ~le, mid, hi)
    tl.store(out + i, lo.to(tl.int32), mask=m)


TYPES = {"values": "*fp64", "bounds": "*fp64", "out": "*i32", "n": "i64", "b": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 4


def run(values, bounds, out, n, b):
    if n == 0:
        return
    kernel[(triton.cdiv(n, CONSTANTS["BLOCK"]),)](
        values, bounds, out, n, b, **CONSTANTS, num_warps=NUM_WARPS
    )
