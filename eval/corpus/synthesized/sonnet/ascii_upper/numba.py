import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


def kernel(offsets, chars, out, n):
    i = cuda.grid(1)
    if i < offsets[n]:
        c = chars[i]
        if c >= 97 and c <= 122:
            out[i] = c - 32
        else:
            out[i] = c


TYPES = (int32[:], uint8[:], uint8[:], int64)
_jit = cuda.jit(kernel)


def run(offsets, chars, out, n):
    total = chars.shape[0]
    if total > 0:
        _jit[(total + 255) // 256, 256](offsets, chars, out, n)
