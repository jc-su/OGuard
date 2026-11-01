from numba import cuda
import numpy as np


def kernel(values, out, n, w):
    """Compute sliding window maximum."""
    idx = cuda.grid(1)
    if idx < n:
        start = max(0, idx - w + 1)
        end = idx + 1
        max_val = values[start]
        for i in range(start + 1, end):
            if values[i] > max_val:
                max_val = values[i]
        out[idx] = max_val


TYPES = (
    cuda.int64[:],
    cuda.int64[:],
    cuda.int64,
    cuda.int64,
)


def run(values, out, n, w, grid_size=None):
    """Launch the window_max kernel."""
    if grid_size is None:
        grid_size = max(1, (n + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](values, out, n, w)
