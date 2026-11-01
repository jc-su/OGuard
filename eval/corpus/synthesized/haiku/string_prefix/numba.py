from numba import cuda
import numpy as np


def kernel(offsets, chars, out, n, k):
    """Extract the first k bytes of each string."""
    idx = cuda.grid(1)
    if idx < n:
        start = offsets[idx]
        end = offsets[idx + 1]
        length = min(end - start, k)

        # Copy string bytes
        for j in range(length):
            out[idx * k + j] = chars[start + j]

        # Pad with zeros
        for j in range(length, k):
            out[idx * k + j] = 0


TYPES = (
    cuda.int32[:],
    cuda.uint8[:],
    cuda.uint8[:],
    cuda.int64,
    cuda.int64,
)


def run(offsets, chars, out, n, k, grid_size=None):
    """Launch the string_prefix kernel."""
    if grid_size is None:
        grid_size = max(1, (n + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](offsets, chars, out, n, k)
