"""For each key, the number of elements of a sorted int64 array that are less than it.

Each row runs a branchless binary search: the position grows by powers of two,
from the largest power of two not above max(s, 1) down to 1, so every lane runs the same
floor(log2(max(s, 1))) + 1 steps.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, sorted, out, n, s, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    key = tl.load(keys + i, mask=m)
    s64 = tl.full([], s, tl.int64)
    step = tl.full([], 1, tl.int64)
    while step * 2 <= s64:
        step = step * 2
    pos = tl.full((BLOCK,), 0, tl.int64)
    while step > 0:
        probe = pos + step
        ok = m & (probe <= s64)
        x = tl.load(sorted + probe - 1, mask=ok)
        pos = tl.where(ok & (x < key), probe, pos)
        step = step // 2
    tl.store(out + i, pos.to(tl.int32), mask=m)


TYPES = {"keys": "*i64", "sorted": "*i64", "out": "*i32", "n": "i64", "s": "i64"}
CONSTANTS = {"BLOCK": 512}
NUM_WARPS = 4


def run(keys: torch.Tensor, sorted: torch.Tensor, out: torch.Tensor, n: int, s: int) -> None:
    grid = ((n + CONSTANTS["BLOCK"] - 1) // CONSTANTS["BLOCK"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](keys, sorted, out, n, s, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
