from numba import cuda
import numpy as np


def kernel(valid, count, n):
    """Count null bits (0 bits) in a validity bitmap."""
    idx = cuda.grid(1)
    if idx < n:
        byte_idx = idx // 8
        bit_idx = idx % 8
        # Check if bit is 0 (null)
        if byte_idx < len(valid):
            if (valid[byte_idx] & (1 << bit_idx)) == 0:
                cuda.atomic.add(count, 0, 1)


TYPES = (
    cuda.uint8[:],
    cuda.int64[:],
    cuda.int64,
)


def run(valid, count, n, grid_size=None):
    """Launch the null_count kernel."""
    if grid_size is None:
        grid_size = max(1, (n + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](valid, count, n)
