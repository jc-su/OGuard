from numba import cuda
import numpy as np


def kernel(source, out, rows, cols):
    """Transpose a matrix from row-major to column-major."""
    idx = cuda.grid(1)
    if idx < rows * cols:
        i = idx // cols
        j = idx % cols
        # source is row-major: source[i * cols + j]
        # output is row-major: out[j * rows + i]
        out[j * rows + i] = source[i * cols + j]


TYPES = (
    cuda.int32[:],
    cuda.int32[:],
    cuda.int64,
    cuda.int64,
)


def run(source, out, rows, cols, grid_size=None):
    """Launch the transpose kernel."""
    total = rows * cols
    if grid_size is None:
        grid_size = max(1, (total + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](source, out, rows, cols)
