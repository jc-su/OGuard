import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


def kernel(col, idx, out, m, n):
    i = cuda.grid(1)
    if i < m:
        j = idx[i]
        if j < 0:
            out[i] = np.nan
        else:
            out[i] = col[j]


TYPES = (float64[:], int32[:], float64[:], int64, int64)
_jit = cuda.jit(kernel)


def run(col, idx, out, m, n):
    if m > 0:
        _jit[(m + 255) // 256, 256](col, idx, out, m, n)
