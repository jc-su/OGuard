"""histogram: counts[(key >> shift) & 255] += 1, one radix-sort digit pass.

Each program walks tiles of BLOCK keys with a grid stride, accumulates a
private 256-bin histogram with tl.histogram, and flushes its non-zero bins to
the global counts with one atomicAdd per bin per program.

counts must be zero on entry.
"""

import torch
import triton
import triton.language as tl

BLOCK = 4096
BINS = 256
NUM_WARPS = 8
# Programs per SM. A small grid keeps the per-program flush (256 global
# atomics) cheap relative to the tile work.
PROGRAMS_PER_SM = 2

TYPES = {"keys": "*u32", "counts": "*i32", "shift": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "BINS": BINS}


@triton.jit
def kernel(keys, counts, shift: tl.int64, n: tl.int64, BLOCK: tl.constexpr, BINS: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(n, BLOCK)
    lanes = tl.arange(0, BLOCK)
    s = shift.to(tl.uint32)
    hist = tl.zeros([BINS], tl.int32)
    for tile in range(pid, num_tiles, num_programs):
        rows = tile.to(tl.int64) * BLOCK + lanes
        in_bounds = rows < n
        k = tl.load(keys + rows, mask=in_bounds, other=0)
        digit = ((k >> s) & (BINS - 1)).to(tl.int32)
        hist += tl.histogram(digit, BINS, mask=in_bounds)
    bins = tl.arange(0, BINS)
    tl.atomic_add(counts + bins, hist, mask=hist != 0, sem="relaxed")


def run(keys, counts, shift, n):
    max_programs = (
        PROGRAMS_PER_SM * torch.cuda.get_device_properties(keys.device).multi_processor_count
    )
    grid = (max(1, min(triton.cdiv(n, BLOCK), max_programs)),)
    kernel[grid](keys, counts, shift, n, **CONSTANTS, num_warps=NUM_WARPS)
