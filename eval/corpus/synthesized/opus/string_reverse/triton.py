"""Reverse the bytes of each string of an Arrow string column.

One program handles BLOCK consecutive bytes of chars. Two scalar binary searches
over offsets find the strings that hold the program's first and last byte; each
byte then finds its own string by a binary search between those two strings, and
the byte at position p of string [start, end) goes to out[start + end - 1 - p].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _last_le(offsets, lo, hi, key, active):
    # Largest i in [lo, hi] with offsets[i] <= key, given offsets[lo] <= key.
    d = hi - lo
    step = tl.full([], 1, tl.int64)
    while step * 2 <= d:
        step = step * 2
    pos = tl.full(key.shape, 0, tl.int64) + lo
    while step > 0:
        probe = pos + step
        ok = active & (probe <= hi)
        x = tl.load(offsets + probe, mask=ok)
        pos = tl.where(ok & (x <= key), probe, pos)
        step = step // 2
    return pos


@triton.jit
def kernel(offsets, chars, out, n, BLOCK: tl.constexpr):
    n64 = tl.full([], n, tl.int64)
    total = tl.load(offsets + n64).to(tl.int64)
    p0 = tl.program_id(0).to(tl.int64) * BLOCK
    p = p0 + tl.arange(0, BLOCK)
    m = p < total
    plast = tl.minimum(p0 + BLOCK, total) - 1
    zero = tl.full([], 0, tl.int64)
    # ty rejects calls to Triton @jit functions by generic invariance; Triton accepts them.
    a = _last_le(offsets, zero, n64, p0, p0 < total)  # ty: ignore[invalid-argument-type]
    b = _last_le(offsets, a, n64, plast, p0 < total)  # ty: ignore[invalid-argument-type]
    i = _last_le(offsets, a, b, p, m)  # ty: ignore[invalid-argument-type]
    start = tl.load(offsets + i, mask=m).to(tl.int64)
    end = tl.load(offsets + i + 1, mask=m).to(tl.int64)
    c = tl.load(chars + p, mask=m)
    tl.store(out + (start + end - 1 - p), c, mask=m)


TYPES = {"offsets": "*i32", "chars": "*u8", "out": "*u8", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(offsets: torch.Tensor, chars: torch.Tensor, out: torch.Tensor, n: int) -> None:
    # chars holds offsets[n] bytes, so its length sizes the grid.
    grid = ((chars.numel() + CONSTANTS["BLOCK"] - 1) // CONSTANTS["BLOCK"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](offsets, chars, out, n, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
