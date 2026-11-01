import torch
import triton
import triton.language as tl


@triton.jit
def kernel(source, out, rows, cols, TILE: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    j = tl.program_id(1).to(tl.int64) * TILE + tl.arange(0, TILE)
    imask = i < rows
    jmask = j < cols
    tile = tl.load(
        source + i[:, None] * cols + j[None, :],
        mask=imask[:, None] & jmask[None, :],
    )
    tl.store(
        out + j[:, None] * rows + i[None, :],
        tl.trans(tile),
        mask=jmask[:, None] & imask[None, :],
    )


TYPES = {"source": "*i32", "out": "*i32", "rows": "i64", "cols": "i64"}
CONSTANTS = {"TILE": 64}
NUM_WARPS = 8


def run(source, out, rows, cols):
    t = CONSTANTS["TILE"]
    grid = (max(1, triton.cdiv(rows, t)), max(1, triton.cdiv(cols, t)))
    kernel[grid](source, out, rows, cols, **CONSTANTS, num_warps=NUM_WARPS)
