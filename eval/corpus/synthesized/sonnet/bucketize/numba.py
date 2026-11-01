import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


def kernel(values, bounds, out, n, b):
    i = cuda.grid(1)
    if i < n:
        v = values[i]
        lo = 0
        hi = b
        while lo < hi:
            mid = (lo + hi) >> 1
            if bounds[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo


TYPES = (float64[:], float64[:], int32[:], int64, int64)
_jit = cuda.jit(kernel)


def run(values, bounds, out, n, b):
    if n > 0:
        _jit[(n + 255) // 256, 256](values, bounds, out, n, b)
