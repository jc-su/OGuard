import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(values, out, totals, n, BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr):
    # Process one tile at a time
    tile_id = tl.program_id(0)
    tile_start = tile_id * TILE_SIZE
    tile_end = tl.minimum(tile_start + TILE_SIZE, n)
    tile_len = tile_end - tile_start

    # Process this tile in blocks
    cum_sum = tl.cast(0, tl.int32)

    for block_id in range(0, TILE_SIZE, BLOCK_SIZE):
        block_offsets_in_tile = block_id + tl.arange(0, BLOCK_SIZE)
        offsets = tile_start + block_offsets_in_tile

        mask = offsets < tile_end

        # Load values
        vals = tl.load(values + offsets, mask=mask, other=0)

        # Compute prefix sum within this block
        # cum_sum is the sum from start of tile to start of this block
        block_cum_sum = cum_sum + tl.cumsum(vals, axis=0)

        # Store to output array
        tl.store(out + offsets, block_cum_sum, mask=mask)

        # Update cum_sum for next block
        # Get the last value (sum of entire tile so far)
        block_sum = tl.sum(tl.where(mask, vals, 0))
        cum_sum = cum_sum + block_sum

    # Store tile total (from first block only)
    if tl.program_id(1) == 0:
        tl.store(totals + tile_id, cum_sum)


TYPES = {
    "values": "*i32",
    "out": "*i32",
    "totals": "*i32",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
    "TILE_SIZE": 1024,
}

NUM_WARPS = 4


def run(values, out, totals, n):
    num_tiles = (n + 1023) // 1024
    grid = (num_tiles, 1)
    kernel[grid](values, out, totals, n, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], TILE_SIZE=CONSTANTS["TILE_SIZE"], num_warps=NUM_WARPS)
