from numba import cuda
import numpy as np


def kernel(lengths, offsets, n):
    """Compute prefix sum of lengths."""
    idx = cuda.grid(1)
    if idx <= n:
        if idx == 0:
            offsets[0] = 0
        else:
            # Compute prefix sum up to idx
            total = 0
            for i in range(idx):
                total += lengths[i]
            offsets[idx] = total


TYPES = (
    cuda.int32[:],
    cuda.int32[:],
    cuda.int64,
)


def run(lengths, offsets, n, grid_size=None):
    """Launch the prefix_offsets kernel."""
    if grid_size is None:
        grid_size = max(1, (n + 1 + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](lengths, offsets, n)
