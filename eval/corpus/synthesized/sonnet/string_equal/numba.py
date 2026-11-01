import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


def kernel(a_offsets, a_chars, b_offsets, b_chars, match, n):
    i = cuda.grid(1)
    if i < n:
        a0 = a_offsets[i]
        b0 = b_offsets[i]
        length = a_offsets[i + 1] - a0
        eq = length == b_offsets[i + 1] - b0
        if eq:
            for j in range(length):
                if a_chars[a0 + j] != b_chars[b0 + j]:
                    eq = False
                    break
        if eq:
            match[i] = 1
        else:
            match[i] = 0


TYPES = (int32[:], uint8[:], int32[:], uint8[:], int8[:], int64)
_jit = cuda.jit(kernel)


def run(a_offsets, a_chars, b_offsets, b_chars, match, n):
    if n > 0:
        _jit[(n + 255) // 256, 256](a_offsets, a_chars, b_offsets, b_chars, match, n)
