"""projection: out[i] = price[i] * (1 - discount[i]). Grid-stride loop, one row per iteration."""

from numba import cuda, float64, int64

THREADS = 256
MAX_BLOCKS = 2048


def kernel(price, discount, out, n):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        out[i] = price[i] * (1.0 - discount[i])
        i += stride


TYPES = (float64[:], float64[:], float64[:], int64)

_kernel = cuda.jit(kernel)


def run(price, discount, out, n):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](price, discount, out, n)
