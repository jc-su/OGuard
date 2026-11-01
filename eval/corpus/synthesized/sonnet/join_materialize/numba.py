import numpy as np
from numba import cuda
from numba.types import int32, int64


def kernel(build, probe, offsets, out_build, out_probe, b, p):
    i = np.int64(cuda.blockIdx.x) * 256 + np.int64(cuda.threadIdx.x)
    lane = cuda.threadIdx.x & 31

    base = np.int32(0)
    count = np.int32(0)
    first = np.int32(0)
    if i < p:
        base = offsets[i]
        count = offsets[i + 1] - base
        if count > 0:
            key = probe[i]
            lo = np.int64(0)
            hi = np.int64(b)  # first build row not less than key
            while lo < hi:
                mid = lo + ((hi - lo) >> 1)
                if build[mid] < key:
                    lo = mid + 1
                else:
                    hi = mid
            first = np.int32(lo)

    # A row with at most 32 matches writes them itself. The lanes of the warp then write the
    # matches of each remaining row together, one row at a time. Every lane reaches the ballot.
    if count <= 32:
        for j in range(count):
            out_build[base + j] = first + j
            out_probe[base + j] = np.int32(i)

    pending = np.int64(cuda.ballot_sync(0xFFFFFFFF, count > 32)) & 0xFFFFFFFF
    while pending != 0:
        leader = cuda.ffs(pending) - 1
        lbase = cuda.shfl_sync(0xFFFFFFFF, base, leader)
        lcount = cuda.shfl_sync(0xFFFFFFFF, count, leader)
        lfirst = cuda.shfl_sync(0xFFFFFFFF, first, leader)
        lrow = cuda.shfl_sync(0xFFFFFFFF, np.int32(i), leader)
        j = np.int32(lane)
        while j < lcount:
            out_build[lbase + j] = lfirst + j
            out_probe[lbase + j] = lrow
            j += 32
        pending &= pending - 1


TYPES = (int64[:], int64[:], int32[:], int32[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(build, probe, offsets, out_build, out_probe, b, p):
    grid = max(1, (p + 255) // 256)
    _kernel[grid, 256](build, probe, offsets, out_build, out_probe, b, p)
