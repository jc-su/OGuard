"""Numba CUDA kernel for filter_select.

Writes the row ids i where quantity[i] < threshold into sel, in any
order, and the number of matches into count.

Precondition: the caller zeroes count before each launch. This matches
the "zero on entry" convention used for the accumulator outputs in the
other tasks in this corpus.
"""

from numba import cuda, float64, int32, int64

TYPES = (float64[:], int32[:], int32[:], float64, int64)

BLOCK = 256


def kernel(quantity, sel, count, threshold, n):
    i = cuda.grid(1)
    if i < n and quantity[i] < threshold:
        pos = cuda.atomic.add(count, 0, 1)
        sel[pos] = i


def run(quantity, sel, count, threshold, n):
    """Launch kernel on device arrays. count must already be zeroed."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](quantity, sel, count, threshold, n)
