"""rolling_sum: trailing window sum by a block-local scan of differences.

A program owns BLOCK output rows and loads a tile of TILE = BLOCK + WMAX
values that starts WMAX rows earlier (the halo covers the largest window). For
tile position t, z[t] = x[t] - x[t - window] (the subtracted term is zero for
t < window), and the inclusive scan of z at t is exactly the sum of the window
ending at t. Every partial sum of the scan is itself a window sum, so no large
prefix sums are subtracted from each other. Rows before the first row read as
zero, which gives the shorter windows at the start of the column.
"""

import torch
import triton
import triton.language as tl

WMAX = 1024
BLOCK = 3072
NUM_WARPS = 4
TYPES = {"values": "*fp64", "out": "*fp64", "n": "i64", "window": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "WMAX": WMAX}


@triton.jit
def kernel(values, out, n: tl.int64, window: tl.int64, BLOCK: tl.constexpr, WMAX: tl.constexpr):
    TILE: tl.constexpr = BLOCK + WMAX
    start = tl.program_id(0).to(tl.int64) * BLOCK
    t = tl.arange(0, TILE)
    g = start - WMAX + t
    x = tl.load(values + g, mask=(g >= 0) & (g < n), other=0.0)
    gy = g - window
    y = tl.load(values + gy, mask=(t >= window) & (gy >= 0) & (gy < n), other=0.0)
    s = tl.cumsum(x - y, axis=0)
    tl.store(out + g, s, mask=(t >= WMAX) & (g < n))


def run(values: torch.Tensor, out: torch.Tensor, n: int, window: int) -> None:
    if not 1 <= window <= WMAX:
        raise ValueError(f"window must be in [1, {WMAX}], got {window}")
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](values, out, n, window, BLOCK=BLOCK, WMAX=WMAX, num_warps=NUM_WARPS)
