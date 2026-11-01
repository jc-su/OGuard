"""Sort int32 keys ascending within tiles of 1024 consecutive rows.

Each program sorts one tile with the bitonic sort of tl.sort. The
rows past the end of a short last tile are padded with INT32_MAX, which sort
after every key, so the first rows of the sorted tile are the tile's keys.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, out, n, TILE: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    in_range = offs < n
    x = tl.load(keys + offs, mask=in_range, other=2147483647)
    tl.store(out + offs, tl.sort(x), mask=in_range)


TYPES: dict[str, str] = {
    "keys": "*i32",
    "out": "*i32",
    "n": "i64",
}
CONSTANTS: dict[str, int] = {"TILE": 1024}
NUM_WARPS: int = 1


def run(keys: torch.Tensor, out: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["TILE"]),)
    kernel[grid](keys, out, n, **CONSTANTS, num_warps=NUM_WARPS)
