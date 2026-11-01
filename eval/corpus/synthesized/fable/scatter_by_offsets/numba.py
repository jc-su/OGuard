"""scatter_by_offsets: out[offsets[i]] = i for every i with flags[i] != 0.

offsets is the exclusive prefix sum of flags, so the selected rows land in
increasing row order. Grid-stride loop, one row per iteration.
"""

from numba import cuda, int8, int32, int64

THREADS = 256
MAX_BLOCKS = 2048


def kernel(flags, offsets, out, n, total):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        if flags[i] != 0:
            out[offsets[i]] = i
        i += stride


TYPES = (int8[:], int32[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(flags, offsets, out, n, total):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](flags, offsets, out, n, total)
