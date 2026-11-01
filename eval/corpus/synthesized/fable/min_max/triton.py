"""min_max: register reduction over TILES tiles per program, then one fp64
atomic min and one atomic max per program on the two result cells.

Each program reduces BLOCK * TILES contiguous values; the tile loop is
unrolled so the loads of all tiles are in flight together.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
TILES = 8
NUM_WARPS = 4
TYPES = {"values": "*fp64", "result": "*fp64", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "TILES": TILES}


@triton.jit
def kernel(values, result, n: tl.int64, BLOCK: tl.constexpr, TILES: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * (BLOCK * TILES)
    ar = tl.arange(0, BLOCK)
    lo = tl.full([BLOCK], float("inf"), tl.float64)
    hi = tl.full([BLOCK], float("-inf"), tl.float64)
    for t in tl.static_range(TILES):
        i = base + t * BLOCK + ar
        m = i < n
        x = tl.load(values + i, mask=m, other=0.0)
        lo = tl.minimum(lo, tl.where(m, x, float("inf")))
        hi = tl.maximum(hi, tl.where(m, x, float("-inf")))
    tl.atomic_min(result, tl.min(lo, axis=0), sem="relaxed")
    tl.atomic_max(result + 1, tl.max(hi, axis=0), sem="relaxed")


def run(values: torch.Tensor, result: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, BLOCK * TILES),)
    kernel[grid](values, result, n, BLOCK=BLOCK, TILES=TILES, num_warps=NUM_WARPS)
