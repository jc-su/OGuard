import numpy as np
from numba import cuda, int64

THREADS = 256


def kernel(ts, out, n, unit):
    i = np.int64(cuda.blockIdx.x) * 256 + cuda.threadIdx.x
    if i < n:
        t = ts[i]
        r = t % unit
        if r < 0:
            r += unit
        out[i] = t - r


TYPES = (int64[:], int64[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(ts, out, n, unit):
    grid = max(1, (n + THREADS - 1) // THREADS)
    _kernel[grid, THREADS](ts, out, n, unit)
