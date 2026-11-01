import numpy as np
from numba import cuda, int32, int64

BLOCK = 256
INT64_MIN = np.int64(np.iinfo(np.int64).min)


def kernel(offsets, values, maxes, g):
    # One warp per segment, with warps taking segments in a grid-stride loop.
    lane = cuda.threadIdx.x & 31
    warp = (int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x) >> 5
    nwarps = (int64(cuda.gridDim.x) * cuda.blockDim.x) >> 5
    for j in range(warp, g, nwarps):
        m = INT64_MIN
        for k in range(int64(offsets[j]) + lane, int64(offsets[j + 1]), 32):
            v = values[k]
            if v > m:
                m = v
        d = 16
        while d > 0:
            other = cuda.shfl_down_sync(0xFFFFFFFF, m, d)
            if other > m:
                m = other
            d >>= 1
        if lane == 0:
            maxes[j] = m


TYPES = (int32[:], int64[:], int64[:], int64)


def run(offsets, values, maxes, g):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min(-(-g // (BLOCK // 32)), 8 * sms))
    cuda.jit(kernel)[blocks, BLOCK](offsets, values, maxes, g)
