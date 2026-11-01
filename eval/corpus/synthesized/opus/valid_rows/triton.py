"""Stream compaction of an Arrow validity bitmap into the ids of the valid rows.

Each program owns BYTES bitmap bytes (8 * BYTES rows). It counts the valid rows
of every byte with a popcount, scans the byte counts, reserves the tile's output
slots with one atomic add on `count`, and then writes each valid row id at
tile base + byte prefix + popcount of the lower bits of its byte. The per-row
work runs on a [BYTES, 8] tile whose lanes map to consecutive rows, so the
stores of a warp cover a contiguous output range. Row ids are ascending inside
a tile; tiles are placed in the order their atomics complete.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def kernel(valid, rows, count, n, BYTES: tl.constexpr):
    pid = tl.program_id(0)
    row0 = pid.to(tl.int64) * (8 * BYTES)
    tile_rows = tl.minimum(n - row0, 8 * BYTES).to(tl.int32)
    local_byte = tl.arange(0, BYTES)
    bits = tl.load(
        valid + pid.to(tl.int64) * BYTES + local_byte,
        mask=local_byte * 8 < tile_rows,
        other=0,
    ).to(tl.int32)
    # Arrow leaves the padding bits past row n - 1 unspecified: clear them.
    rows_in_byte = tl.minimum(tl.maximum(tile_rows - local_byte * 8, 0), 8)
    bits = bits & ((1 << rows_in_byte) - 1)
    per_byte = libdevice.popc(bits)
    byte_start = tl.cumsum(per_byte, axis=0) - per_byte
    total = tl.sum(per_byte, axis=0)
    base = tl.atomic_add(count, total, mask=total > 0, sem="relaxed")
    bit = tl.arange(0, 8)
    is_valid = ((bits[:, None] >> bit[None, :]) & 1) != 0
    lower_valid = libdevice.popc(bits[:, None] & ((1 << bit[None, :]) - 1))
    pos = (base + byte_start)[:, None] + lower_valid
    row = row0.to(tl.int32) + local_byte[:, None] * 8 + bit[None, :]
    tl.store(rows + pos, row, mask=is_valid)


TYPES = {"valid": "*u8", "rows": "*i32", "count": "*i32", "n": "i64"}
CONSTANTS = {"BYTES": 512}
NUM_WARPS = 4


def run(valid: torch.Tensor, rows: torch.Tensor, count: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, 8 * CONSTANTS["BYTES"]),)
    kernel[grid](valid, rows, count, n, num_warps=NUM_WARPS, **CONSTANTS)
