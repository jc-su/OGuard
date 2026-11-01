"""Stream compaction of an Arrow validity bitmap into the ids of the valid rows.

Each program owns BYTES bitmap bytes (8 * BYTES rows). It counts the valid rows
of every byte with a popcount, scans the byte counts, reserves the tile's output
slots with one atomic add on `count`, and then writes each valid row id at
tile base + byte prefix + popcount of the lower bits of its byte. The per-row
work runs on a [BYTES, 8] tile whose lanes map to consecutive rows, so the
stores of a warp cover a contiguous output range. Row ids are ascending inside
a tile; tiles are placed in the order their atomics complete.

Bounds: each masked access has a mask that is one comparison of its own index.
A byte is read when it is below the bitmap length (n + 7) >> 3. A store slot
comes from the atomic on `count`; lanes without a valid row get the slot n, and
the store mask is slot < n. With `count` zero on entry every valid row's slot
is in [0, n), so the mask never drops a row.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def kernel(valid, rows, count, n, BYTES: tl.constexpr):
    nbytes = (n + 7) >> 3  # the length of the bitmap
    byte = tl.program_id(0).to(tl.int64) * BYTES + tl.arange(0, BYTES)
    bits = tl.load(valid + byte, mask=byte < nbytes, other=0).to(tl.int32)
    row0 = byte * 8  # the row of bit 0 of each byte
    # Arrow leaves the padding bits past row n - 1 unspecified: clear them.
    rows_in_byte = tl.minimum(tl.maximum(n - row0, 0), 8).to(tl.int32)
    bits = bits & ((1 << rows_in_byte) - 1)
    per_byte = libdevice.popc(bits)
    byte_start = tl.cumsum(per_byte, axis=0) - per_byte
    total = tl.sum(per_byte, axis=0)
    base = tl.atomic_add(count, total, mask=total > 0, sem="relaxed")
    bit = tl.arange(0, 8)
    is_valid = ((bits[:, None] >> bit[None, :]) & 1) != 0
    lower_valid = libdevice.popc(bits[:, None] & ((1 << bit[None, :]) - 1))
    pos = (base + byte_start).to(tl.int64)[:, None] + lower_valid
    slot = tl.where(is_valid, tl.maximum(pos, 0), n)
    row = (row0[:, None] + bit[None, :]).to(tl.int32)
    tl.store(rows + slot, row, mask=slot < n)


TYPES = {"valid": "*u8", "rows": "*i32", "count": "*i32", "n": "i64"}
CONSTANTS = {"BYTES": 512}
NUM_WARPS = 4


def run(valid: torch.Tensor, rows: torch.Tensor, count: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, 8 * CONSTANTS["BYTES"]),)
    kernel[grid](valid, rows, count, n, num_warps=NUM_WARPS, **CONSTANTS)
