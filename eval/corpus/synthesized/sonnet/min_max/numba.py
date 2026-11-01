"""Numba CUDA kernel: min_max.

Reduces `values` to its minimum and maximum, merged into result[0] and
result[1] (initialized to +inf / -inf by the caller). Each block reduces
its own tile in shared memory (a standard tree reduction) and only the
block's single winning min and single winning max are atomically merged
into the global result, so the number of atomic operations is
ceil(n / 256) rather than n.
"""

import math

from numba import cuda, float64, int64

TYPES = (float64[:], float64[:], int64)

THREADS_PER_BLOCK = 256


def kernel(values, result, n):
    smin = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)
    smax = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)

    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    if i < n:
        v = values[i]
        smin[tid] = v
        smax[tid] = v
    else:
        smin[tid] = float64(math.inf)
        smax[tid] = float64(-math.inf)
    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            if smin[tid + s] < smin[tid]:
                smin[tid] = smin[tid + s]
            if smax[tid + s] > smax[tid]:
                smax[tid] = smax[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        cuda.atomic.min(result, 0, smin[0])
        cuda.atomic.max(result, 1, smax[0])


def run(values, result, n):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](values, result, n)
