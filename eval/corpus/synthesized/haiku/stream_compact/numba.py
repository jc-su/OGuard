from numba import cuda
import numpy as np


def kernel(values, keep, out, count, n):
    """Stream compact kernel using atomic operations."""
    idx = cuda.grid(1)
    if idx < n:
        if keep[idx] != 0:
            pos = cuda.atomic.add(count, 0, 1)
            out[pos] = values[idx]


TYPES = (
    cuda.int64[:],
    cuda.int8[:],
    cuda.int64[:],
    cuda.int32[:],
    cuda.int64,
)


def run(values, keep, out, count, n, grid_size=None):
    """Launch the stream_compact kernel."""
    if grid_size is None:
        grid_size = max(1, (n + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](values, keep, out, count, n)
