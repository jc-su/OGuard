"""Reverse each tile of 1024 consecutive int32 rows; the last tile can be shorter.

One program handles one tile.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, out, n, TILE: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * TILE
    length = tl.minimum(n - base, TILE)
    j = tl.arange(0, TILE)
    m = j < length
    v = tl.load(values + base + j, mask=m)
    tl.store(out + base + (length - 1 - j), v, mask=m)


TYPES = {"values": "*i32", "out": "*i32", "n": "i64"}
CONSTANTS = {"TILE": 1024}
NUM_WARPS = 4


def run(values: torch.Tensor, out: torch.Tensor, n: int) -> None:
    grid = ((n + CONSTANTS["TILE"] - 1) // CONSTANTS["TILE"],)
    # Triton types the launch as the kernel's own signature: it rejects the num_warps
    # launch option and ints for the tl.constexpr parameters, which Triton accepts.
    kernel[grid](values, out, n, **CONSTANTS, num_warps=NUM_WARPS)  # ty: ignore[invalid-argument-type, unknown-argument]
