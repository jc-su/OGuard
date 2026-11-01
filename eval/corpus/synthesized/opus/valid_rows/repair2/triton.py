"""Stream compaction of an Arrow validity bitmap into the ids of the valid rows.

Each program owns BLOCK consecutive rows. Every row reads its bitmap byte
row >> 3 and its bit row & 7; an exclusive scan of the bits gives each valid
row its place among the tile's valid rows, and one atomic add of the tile's
count on `count` reserves the tile's output slots. A valid row then goes to
the slot tile base + place, so the stores of a warp cover a contiguous output
range. Row ids are ascending inside a tile; tiles are placed in the order
their atomics complete.

Bounds: every load and store has a mask that holds 0 <= index and
index < length for the very value that bounds it, so its bound needs no
minimum, maximum or loop invariant. A bitmap byte row >> 3 is read under
0 <= row and row < n for the row itself (the bitmap has (n + 7) / 8 bytes, so
row >> 3 is one of them) and under byte < (n + 7) >> 3. A store slot comes
from the atomic on `count` and is stored under 0 <= slot and slot < n; with
`count` zero on entry every valid row's slot is in [0, n), so the mask never
drops a row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(valid, rows, count, n, BLOCK: tl.constexpr):
    nbytes = (n + 7) >> 3  # the length of the bitmap
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    byte = row >> 3
    readable = (row >= 0) & (row < n) & (byte < nbytes)
    bits = tl.load(valid + byte, mask=readable, other=0).to(tl.int32)
    bit = (bits >> (row & 7).to(tl.int32)) & 1
    place = tl.cumsum(bit, axis=0) - bit
    total = tl.sum(bit, axis=0)
    base = tl.atomic_add(count, total, mask=total > 0, sem="relaxed")
    slot = (base + place).to(tl.int64)
    tl.store(rows + slot, row.to(tl.int32), mask=(bit != 0) & (slot >= 0) & (slot < n))


TYPES = {"valid": "*u8", "rows": "*i32", "count": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 4096}
NUM_WARPS = 4


def run(valid: torch.Tensor, rows: torch.Tensor, count: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](valid, rows, count, n, num_warps=NUM_WARPS, **CONSTANTS)
