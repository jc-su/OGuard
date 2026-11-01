"""Transpose of a row-major int32 matrix, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "cols"])
def kernel(source, out, rows, cols, TILE: tl.constexpr):
    # One program per TILE x TILE tile. The load is coalesced along the columns
    # of source and, after tl.trans, the store is coalesced along the columns
    # of out.
    pid = tl.program_id(0).to(tl.int64)
    tiles_across = tl.cdiv(cols.to(tl.int64), TILE)
    tile_row = pid // tiles_across
    tile_col = pid - tile_row * tiles_across
    i = tile_row * TILE + tl.arange(0, TILE)
    j = tile_col * TILE + tl.arange(0, TILE)
    x = tl.load(
        source + i[:, None] * cols + j[None, :],
        mask=(i[:, None] < rows) & (j[None, :] < cols),
    )
    tl.store(
        out + j[:, None] * rows + i[None, :],
        tl.trans(x),
        mask=(j[:, None] < cols) & (i[None, :] < rows),
    )


TYPES: dict[str, str] = {"source": "*i32", "out": "*i32", "rows": "i64", "cols": "i64"}
CONSTANTS: dict[str, int] = {"TILE": 16}
NUM_WARPS: int = 2


def run(source: torch.Tensor, out: torch.Tensor, rows: int, cols: int) -> None:
    tile = CONSTANTS["TILE"]
    grid = (triton.cdiv(rows, tile) * triton.cdiv(cols, tile),)
    kernel[grid](source, out, rows, cols, **CONSTANTS, num_warps=NUM_WARPS)
