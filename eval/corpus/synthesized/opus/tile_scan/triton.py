"""Exclusive prefix sum within tiles of 1024 int32 rows, with tile totals.

out[i] is the sum of values[j] over the rows j < i of row i's tile, and
totals[t] is the sum of tile t. The last tile can be shorter than 1024 rows.
Each program scans one tile.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, out, totals, n, TILE: tl.constexpr):
    tile = tl.program_id(0).to(tl.int64)
    offs = tile * TILE + tl.arange(0, TILE)
    in_range = offs < n
    x = tl.load(values + offs, mask=in_range, other=0)
    inclusive = tl.cumsum(x, axis=0)
    tl.store(out + offs, inclusive - x, mask=in_range)
    tl.store(totals + tile, tl.sum(x, axis=0))


TYPES: dict[str, str] = {
    "values": "*i32",
    "out": "*i32",
    "totals": "*i32",
    "n": "i64",
}
CONSTANTS: dict[str, int] = {"TILE": 1024}
NUM_WARPS: int = 4


def run(values: torch.Tensor, out: torch.Tensor, totals: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["TILE"]),)
    kernel[grid](values, out, totals, n, **CONSTANTS, num_warps=NUM_WARPS)
