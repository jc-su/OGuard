"""gather: out[i] = col[sel[i]] for i in [0, m). Grid-stride loop, one row per iteration."""

from numba import cuda, float64, int32, int64

THREADS = 256
MAX_BLOCKS = 2048


def kernel(col, sel, out, m, n):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < m:
        out[i] = col[sel[i]]
        i += stride


TYPES = (float64[:], int32[:], float64[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(col, sel, out, m, n):
    blocks = max(1, min((m + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](col, sel, out, m, n)
