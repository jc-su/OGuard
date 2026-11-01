"""Numba CUDA kernel for gather.

out[i] = col[sel[i]] for i in [0, m). Every value in sel is a row id
of col, in [0, n).
"""

from numba import cuda, float64, int32, int64

TYPES = (float64[:], int32[:], float64[:], int64, int64)

BLOCK = 256


def kernel(col, sel, out, m, n):
    i = cuda.grid(1)
    if i < m:
        out[i] = col[sel[i]]


def run(col, sel, out, m, n):
    """Launch kernel on device arrays."""
    blocks = (m + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](col, sel, out, m, n)
