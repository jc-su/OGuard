"""min_max: the minimum and maximum of a float64 column (Numba CUDA).

A single-pass reduction. run launches a persistent grid (as many blocks as
can be resident at once, never more than one per 256 rows); each thread folds
its grid-stride share of the rows into a running min and max, the block
combines them with warp shuffles and shared memory, and thread 0 of each block
applies them to result with one atomic min and one atomic max.
"""

import functools
import math

import numpy as np
from numba import cuda, float64, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
WARPS = BLOCK // 32
FULL_MASK = 0xFFFFFFFF

TYPES = (float64[:], float64[:], int64)


def kernel(values: np.ndarray, result: np.ndarray, n: int) -> None:
    warp_min = cuda.shared.array(WARPS, float64)
    warp_max = cuda.shared.array(WARPS, float64)

    lo = math.inf
    hi = -math.inf
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        v = values[i]
        lo = min(lo, v)
        hi = max(hi, v)
        i += stride

    offset = 16
    while offset > 0:
        lo = min(lo, cuda.shfl_down_sync(FULL_MASK, lo, offset))
        hi = max(hi, cuda.shfl_down_sync(FULL_MASK, hi, offset))
        offset >>= 1

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    if lane == 0:
        warp_min[warp] = lo
        warp_max[warp] = hi
    cuda.syncthreads()

    if tid == 0:
        for w in range(1, WARPS):
            lo = min(lo, warp_min[w])
            hi = max(hi, warp_max[w])
        cuda.atomic.min(result, 0, lo)
        cuda.atomic.max(result, 1, hi)


_kernel = cuda.jit(kernel)


@functools.cache
def _resident_blocks() -> int:
    compiled = _kernel.compile((float64[::1], float64[::1], int64))
    return compiled.max_cooperative_grid_blocks(BLOCK)


def run(values: DeviceNDArray, result: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    grid = min((n + BLOCK - 1) // BLOCK, _resident_blocks())
    _kernel[grid, BLOCK](values, result, n)
