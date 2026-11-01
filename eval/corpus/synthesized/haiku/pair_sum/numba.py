from numba import cuda
import numpy as np


def kernel(left_idx, right_idx, left, right, out, m, l, r):
    """Sum pairs from left and right arrays."""
    idx = cuda.grid(1)
    if idx < m:
        out[idx] = left[left_idx[idx]] + right[right_idx[idx]]


TYPES = (
    cuda.int32[:],
    cuda.int32[:],
    cuda.float64[:],
    cuda.float64[:],
    cuda.float64[:],
    cuda.int64,
    cuda.int64,
    cuda.int64,
)


def run(left_idx, right_idx, left, right, out, m, l, r, grid_size=None):
    """Launch the pair_sum kernel."""
    if grid_size is None:
        grid_size = max(1, (m + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](left_idx, right_idx, left, right, out, m, l, r)
