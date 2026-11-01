"""Numba CUDA kernel for scatter_by_offsets.

For each i with flags[i] != 0, writes out[offsets[i]] = i. offsets is
the exclusive prefix sum of flags, so distinct selected rows land at
distinct, already-known positions and no atomics are needed.

offsets is an input array, not a scalar with a declared value range,
so the checker cannot take "each used value lands in [0, total)" for
granted from the array's dtype alone: the store is guarded by an
explicit two-sided bounds check on the position read from offsets
before it runs. Under the stated contract, offsets is the true
exclusive prefix sum of flags, so the guard never rejects a position
for a row with flags[i] != 0.
"""

from numba import cuda, int8, int32, int64

TYPES = (int8[:], int32[:], int32[:], int64, int64)

BLOCK = 256


def kernel(flags, offsets, out, n, total):
    i = cuda.grid(1)
    if i < n and flags[i] != 0:
        pos = offsets[i]
        if pos >= 0 and pos < total:
            out[pos] = i


def run(flags, offsets, out, n, total):
    """Launch kernel on device arrays."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](flags, offsets, out, n, total)
