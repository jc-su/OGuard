"""Numba CUDA kernel for segmented_sum.

sums[j] = sum(values[offsets[j]:offsets[j+1]]) for j in [0, g). One
warp is assigned per segment (the CSR-vector pattern): the lanes of
the warp stride over the segment's values, then combine their partial
sums with a warp shuffle reduction. This keeps segments of very
different sizes efficient without one thread, or one block, per
segment.
"""

from numba import cuda, float64, int32, int64

TYPES = (int32[:], float64[:], float64[:], int64)

BLOCK = 256
WARP_SIZE = 32
FULL_MASK = 0xFFFFFFFF


def kernel(offsets, values, sums, g):
    warp_id = cuda.grid(1) // WARP_SIZE
    lane = cuda.threadIdx.x % WARP_SIZE

    if warp_id >= g:
        return

    start = offsets[warp_id]
    end = offsets[warp_id + 1]

    total = 0.0
    idx = start + lane
    while idx < end:
        total += values[idx]
        idx += WARP_SIZE

    offset = WARP_SIZE // 2
    while offset > 0:
        total += cuda.shfl_down_sync(FULL_MASK, total, offset)
        offset //= 2

    if lane == 0:
        sums[warp_id] = total


def run(offsets, values, sums, g):
    """Launch kernel on device arrays."""
    if g == 0:
        return
    threads_needed = g * WARP_SIZE
    blocks = (threads_needed + BLOCK - 1) // BLOCK
    cuda.jit(kernel)[blocks, BLOCK](offsets, values, sums, g)
