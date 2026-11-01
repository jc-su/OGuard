"""valid_rows: compact the ids of the valid rows of an Arrow validity bitmap.

Each program handles BLOCK rows. It loads the BLOCK / 8 bitmap bytes of its
rows, expands them to one bit per row, reserves a segment of ``rows`` with one
atomic add on ``count``, and scatters its valid row ids into that segment at
the offsets given by a block-local exclusive prefix sum.
"""

import torch
import triton
import triton.language as tl

BLOCK = 2048
NUM_WARPS = 4
TYPES = {"valid": "*u8", "rows": "*i32", "count": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(valid, rows, count, n: tl.int64, BLOCK: tl.constexpr):
    start = tl.program_id(0).to(tl.int64) * BLOCK
    byte_idx = start // 8 + tl.arange(0, BLOCK // 8)
    nbytes = (n + 7) // 8
    bytes_ = tl.load(valid + byte_idx, mask=byte_idx < nbytes, other=0)
    # bit j of byte b is row 8 * b + j; a row-major reshape keeps that order
    bits = (bytes_[:, None] >> tl.arange(0, 8)[None, :]) & 1
    bit = tl.reshape(bits, [BLOCK])
    i = start + tl.arange(0, BLOCK)
    v = ((bit != 0) & (i < n)).to(tl.int32)
    total = tl.sum(v, axis=0)
    base = tl.atomic_add(count, total, sem="relaxed")
    pos = base + tl.cumsum(v, axis=0) - v
    tl.store(rows + pos, i.to(tl.int32), mask=v != 0)


def run(valid: torch.Tensor, rows: torch.Tensor, count: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](valid, rows, count, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
