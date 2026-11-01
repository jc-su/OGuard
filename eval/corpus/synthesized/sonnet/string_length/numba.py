"""Numba CUDA kernel for string_length.

Counts the UTF-8 code points of each string i, delimited by
offsets[i]:offsets[i+1] in chars. A byte starts a code point unless it
is a continuation byte (its top two bits are 10), so counting
non-continuation bytes counts code points.
"""

from numba import cuda, int32, int64, uint8

TYPES = (int32[:], uint8[:], int32[:], int64)

BLOCK = 256


def kernel(offsets, chars, lengths, n):
    i = cuda.grid(1)
    if i < n:
        start = offsets[i]
        end = offsets[i + 1]
        count = 0
        for j in range(start, end):
            if (chars[j] & 0xC0) != 0x80:
                count += 1
        lengths[i] = count


def run(offsets, chars, lengths, n):
    """Launch kernel on device arrays."""
    blocks = (n + BLOCK - 1) // BLOCK
    if blocks == 0:
        return
    cuda.jit(kernel)[blocks, BLOCK](offsets, chars, lengths, n)
