import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(keys, out, n, BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr):
    # Process one tile at a time
    tile_id = tl.program_id(0)
    tile_start = tile_id * TILE_SIZE
    tile_end = tl.minimum(tile_start + TILE_SIZE, n)

    # Copy input to output
    for i in range(TILE_SIZE):
        idx = tile_start + i
        if idx < tile_end:
            val = tl.load(keys + idx)
            tl.store(out + idx, val)

    # Odd-even transposition sort
    # Each iteration compares all even pairs, then all odd pairs
    tile_len = tile_end - tile_start

    for iteration in range(TILE_SIZE):
        # Even phase: compare pairs (0,1), (2,3), (4,5), ...
        for i in range(0, TILE_SIZE - 1, 2):
            idx_0 = tile_start + i
            idx_1 = tile_start + i + 1

            if idx_0 < tile_end and idx_1 < tile_end:
                val_0 = tl.load(out + idx_0)
                val_1 = tl.load(out + idx_1)

                if val_0 > val_1:
                    tl.store(out + idx_0, val_1)
                    tl.store(out + idx_1, val_0)

        # Odd phase: compare pairs (1,2), (3,4), (5,6), ...
        for i in range(1, TILE_SIZE - 1, 2):
            idx_0 = tile_start + i
            idx_1 = tile_start + i + 1

            if idx_0 < tile_end and idx_1 < tile_end:
                val_0 = tl.load(out + idx_0)
                val_1 = tl.load(out + idx_1)

                if val_0 > val_1:
                    tl.store(out + idx_0, val_1)
                    tl.store(out + idx_1, val_0)


TYPES = {
    "keys": "*i32",
    "out": "*i32",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
    "TILE_SIZE": 1024,
}

NUM_WARPS = 4


def run(keys, out, n):
    num_tiles = (n + 1023) // 1024
    grid = (num_tiles,)
    kernel[grid](keys, out, n, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], TILE_SIZE=CONSTANTS["TILE_SIZE"], num_warps=NUM_WARPS)
