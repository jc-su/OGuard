"""min_max: minimum and maximum of a float64 column without NaN.

Grid-stride loop with a capped grid: each thread folds its elements into a
register min and max, each warp reduces with xor shuffles, the eight warp
results are reduced through shared memory, and thread 0 issues one atomic
min and one atomic max per block. A block that saw no element (n = 0) skips
the atomics, so the result keeps its entry values.
"""

import math

from numba import cuda, float64, int64

TYPES = (float64[:], float64[:], int64)
THREADS = 256
MAX_BLOCKS = 2048


def kernel(values, result, n):
    warp_min = cuda.shared.array(8, float64)
    warp_max = cuda.shared.array(8, float64)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5

    lo = math.inf
    hi = -math.inf
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        v = values[i]
        lo = min(lo, v)
        hi = max(hi, v)
        i += stride

    off = 16
    while off > 0:
        lo = min(lo, cuda.shfl_xor_sync(0xFFFFFFFF, lo, off))
        hi = max(hi, cuda.shfl_xor_sync(0xFFFFFFFF, hi, off))
        off >>= 1
    if lane == 0:
        warp_min[wid] = lo
        warp_max[wid] = hi
    cuda.syncthreads()

    if tid == 0:
        for w in range(1, 8):
            lo = min(lo, warp_min[w])
            hi = max(hi, warp_max[w])
        if lo <= hi:
            cuda.atomic.min(result, 0, lo)
            cuda.atomic.max(result, 1, hi)


_kernel = cuda.jit(kernel)


def run(values, result, n):
    blocks = min(MAX_BLOCKS, max(1, (n + THREADS - 1) // THREADS))
    _kernel[blocks, THREADS](values, result, n)
