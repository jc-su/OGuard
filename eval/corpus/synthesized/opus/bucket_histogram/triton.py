"""Histogram of float64 values over the buckets given by sorted bounds.

Each row finds its bucket c, the number of bounds <= value, by a branchless
binary search. When the b + 1 buckets fit in LOCAL bins, each program builds a
histogram of its BLOCK rows with tl.histogram and adds its nonzero bins with
atomics; otherwise each row adds 1 to its bucket with an atomic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, bounds, counts, n, b, BLOCK: tl.constexpr, LOCAL: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(values + i, mask=m, other=0.0)
    b64 = tl.full([], b, tl.int64)
    step = tl.full([], 1, tl.int64)
    while step * 2 <= b64:
        step = step * 2
    c = tl.full((BLOCK,), 0, tl.int64)
    while step > 0:
        probe = c + step
        ok = m & (probe <= b64)
        x = tl.load(bounds + probe - 1, mask=ok)
        c = tl.where(ok & (x <= v), probe, c)
        step = step // 2
    if b64 < LOCAL:
        h = tl.histogram(c.to(tl.int32), LOCAL, mask=m)
        bins = tl.arange(0, LOCAL)
        tl.atomic_add(counts + bins, h, mask=(bins <= b64) & (h > 0), sem="relaxed")
    else:
        tl.atomic_add(counts + c, tl.full((BLOCK,), 1, tl.int32), mask=m, sem="relaxed")


TYPES = {"values": "*fp64", "bounds": "*fp64", "counts": "*i32", "n": "i64", "b": "i64"}
CONSTANTS = {"BLOCK": 1024, "LOCAL": 256}
NUM_WARPS = 4


def run(values: torch.Tensor, bounds: torch.Tensor, counts: torch.Tensor, n: int, b: int) -> None:
    grid = ((n + CONSTANTS["BLOCK"] - 1) // CONSTANTS["BLOCK"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](values, bounds, counts, n, b, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
