"""Numba CUDA kernel for scatter_by_offsets.

For each i with flags[i] != 0, writes out[offsets[i]] = i. offsets is
the exclusive prefix sum of flags, so distinct selected rows land at
distinct, already-known positions and no atomics are needed.
"""

from numba import cuda, int8, int32, int64

TYPES = (int8[:], int32[:], int32[:], int64, int64)

BLOCK = 256


def kernel(flags, offsets, out, n, total):
    i = cuda.grid(1)
    if i < n and flags[i] != 0:
        out[offsets[i]] = i


def run(flags, offsets, out, n, total):
    """Launch kernel on device arrays."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](flags, offsets, out, n, total)
