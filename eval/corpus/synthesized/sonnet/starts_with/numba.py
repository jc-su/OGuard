"""Numba CUDA kernel for starts_with.

match[i] = 1 when string i (offsets[i]:offsets[i+1] in chars) starts
with the k-byte pattern prefix, else 0. The comparison is byte-wise,
not code-point aware.
"""

from numba import cuda, int8, int32, int64, uint8

TYPES = (int32[:], uint8[:], uint8[:], int8[:], int64, int64)

BLOCK = 256


def kernel(offsets, chars, prefix, match, n, k):
    i = cuda.grid(1)
    if i >= n:
        return

    start = offsets[i]
    end = offsets[i + 1]

    ok = 1
    if end - start < k:
        ok = 0
    else:
        for j in range(k):
            if chars[start + j] != prefix[j]:
                ok = 0
                break
    match[i] = ok


def run(offsets, chars, prefix, match, n, k):
    """Launch kernel on device arrays."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](offsets, chars, prefix, match, n, k)
