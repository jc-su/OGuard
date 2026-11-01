"""Scatter pass of an LSD radix sort on 8-bit digits.

Each program takes BLOCK rows and ranks them locally: it sorts the packed pairs
(digit, local row), so the rows of one digit become contiguous and in row
order, builds the tile's digit histogram, and reserves the output range of
each digit with one vector atomic add of the histogram on `cursors` (one atomic
per digit present, instead of one per row). A row at sorted position q with
digit d then goes to cursor_before[d] + q - tile_digit_start[d]. Rows of one
digit keep their order inside a tile and the stores of a digit are contiguous.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, cursors, out, n, shift, BLOCK: tl.constexpr):
    row0 = tl.program_id(0).to(tl.int64) * BLOCK
    local = tl.arange(0, BLOCK)
    in_range = row0 + local < n
    key = tl.load(keys + row0 + local, mask=in_range, other=0)
    digit = ((key >> tl.cast(shift, tl.uint32)) & 255).to(tl.int32)
    # Rows past n get digit 256 and sort to the end of the tile.
    packed = tl.where(in_range, digit, 256) * BLOCK + local
    ordered = tl.sort(packed)
    ordered_digit = ordered // BLOCK
    ordered_row = ordered % BLOCK
    count = tl.histogram(digit, 256, mask=in_range)
    tile_start = tl.cumsum(count, axis=0) - count
    bins = tl.arange(0, 256)
    cursor = tl.atomic_add(cursors + bins, count, mask=count > 0, sem="relaxed")
    shift_to_output = tl.gather(cursor - tile_start, tl.minimum(ordered_digit, 255), 0)
    tl.store(
        out + (shift_to_output + local),
        (row0 + ordered_row).to(tl.int32),
        mask=ordered_digit < 256,
    )


TYPES = {"keys": "*u32", "cursors": "*i32", "out": "*i32", "n": "i64", "shift": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(keys: torch.Tensor, cursors: torch.Tensor, out: torch.Tensor, n: int, shift: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, cursors, out, n, shift, num_warps=NUM_WARPS, **CONSTANTS)
