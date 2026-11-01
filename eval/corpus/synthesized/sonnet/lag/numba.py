import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


def kernel(values, first, out, n, k):
    i = cuda.grid(1)
    if i < n:
        j = i - k
        if j >= first[i]:
            out[i] = values[j]
        else:
            out[i] = np.nan


TYPES = (float64[:], int32[:], float64[:], int64, int64)
_jit = cuda.jit(kernel)


def run(values, first, out, n, k):
    if n > 0:
        _jit[(n + 255) // 256, 256](values, first, out, n, k)
