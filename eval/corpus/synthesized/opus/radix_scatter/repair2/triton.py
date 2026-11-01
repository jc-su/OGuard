"""Scatter pass of an LSD radix sort on 8-bit digits.

Each program takes BLOCK rows and ranks them locally: it sorts the packed pairs
(digit, local row), so the rows of one digit become contiguous and in row
order, builds the tile's digit histogram, and reserves the output range of
each digit with one vector atomic add of the histogram on `cursors` (one atomic
per digit present, instead of one per row). A row at sorted position q with
digit d then goes to cursor_before[d] + q - tile_digit_start[d]. Rows of one
digit keep their order inside a tile and the stores of a digit are contiguous.

The kernel has no branches: the loads, the atomics and the stores are
predicated. The histogram counts every row of the tile without a mask (a mask
makes Triton guard each shared-memory atomic with a branch); a row past n
reads the key 0, so the rows past n are taken back from bin 0.

A program is one warp (NUM_WARPS = 1): the sort and the scan exchange values
between lanes with warp shuffles, and only the histogram and the gather use
shared memory.

Bounds: a key index is row0 + t with the mask index < n; a cursor index is a
bin in [0, 255]; the digit that selects a bin in the gather is masked with
255. An output slot comes from the atomic on `cursors`, and the store mask
holds the two comparisons 0 <= slot and slot < n of the very slot the store
uses, so no bound needs a minimum or a maximum. For cursors that meet the
contract every slot is in [0, n), so the mask drops only the rows past n.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, cursors, out, n, shift, BLOCK: tl.constexpr):
    row0 = tl.program_id(0).to(tl.int64) * BLOCK
    local = tl.arange(0, BLOCK)
    row = row0 + local
    in_range = row < n
    key = tl.load(keys + row, mask=in_range, other=0)
    digit = ((key >> tl.cast(shift, tl.uint32)) & 255).to(tl.int32)
    # Rows past n get digit 256 and sort to the end of the tile.
    packed = tl.where(in_range, digit, 256) * BLOCK + local
    ordered = tl.sort(packed)
    ordered_digit = ordered // BLOCK
    ordered_row = ordered % BLOCK
    bins = tl.arange(0, 256)
    past_end = tl.sum((~in_range).to(tl.int32), axis=0)
    count = tl.histogram(digit, 256) - tl.where(bins == 0, past_end, 0)
    tile_start = tl.cumsum(count, axis=0) - count
    cursor = tl.atomic_add(cursors + bins, count, mask=count > 0, sem="relaxed")
    shift_to_output = tl.gather(cursor - tile_start, ordered_digit & 255, 0)
    slot = (shift_to_output + local).to(tl.int64)
    stored = (ordered_digit < 256) & (slot >= 0) & (slot < n)
    tl.store(out + slot, (row0 + ordered_row).to(tl.int32), mask=stored)


TYPES = {"keys": "*u32", "cursors": "*i32", "out": "*i32", "n": "i64", "shift": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 1


def run(keys: torch.Tensor, cursors: torch.Tensor, out: torch.Tensor, n: int, shift: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, cursors, out, n, shift, num_warps=NUM_WARPS, **CONSTANTS)
