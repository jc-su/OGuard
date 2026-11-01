"""Numba CUDA kernel for projection.

out[i] = price[i] * (1 - discount[i]) for i in [0, n).
"""

from numba import cuda, float64, int64

TYPES = (float64[:], float64[:], float64[:], int64)

BLOCK = 256


def kernel(price, discount, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = price[i] * (1.0 - discount[i])


def run(price, discount, out, n):
    """Launch kernel on device arrays."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](price, discount, out, n)
