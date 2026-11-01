import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


def kernel(a, valid, b, out, n):
    i = cuda.grid(1)
    if i < n:
        if (valid[i >> 3] >> (i & 7)) & 1:
            out[i] = a[i]
        else:
            out[i] = b[i]


TYPES = (float64[:], uint8[:], float64[:], float64[:], int64)
_jit = cuda.jit(kernel)


def run(a, valid, b, out, n):
    if n > 0:
        _jit[(n + 255) // 256, 256](a, valid, b, out, n)
