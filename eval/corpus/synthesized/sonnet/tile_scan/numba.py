import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


def kernel(values, out, totals, n):
    warp_sums = cuda.shared.array(8, dtype=int64)
    tid = cuda.threadIdx.x
    base = cuda.blockIdx.x * 1024 + tid * 4
    v0 = 0
    v1 = 0
    v2 = 0
    v3 = 0
    if base < n:
        v0 = values[base]
    if base + 1 < n:
        v1 = values[base + 1]
    if base + 2 < n:
        v2 = values[base + 2]
    if base + 3 < n:
        v3 = values[base + 3]
    t = v0 + v1 + v2 + v3
    incl = t
    lane = tid & 31
    d = 1
    while d < 32:
        x = cuda.shfl_up_sync(0xFFFFFFFF, incl, d)
        if lane >= d:
            incl += x
        d <<= 1
    if lane == 31:
        warp_sums[tid >> 5] = incl
    cuda.syncthreads()
    excl = incl - t
    for w in range(tid >> 5):
        excl += warp_sums[w]
    if tid == 0:
        total = 0
        for w in range(8):
            total += warp_sums[w]
        totals[cuda.blockIdx.x] = total
    if base < n:
        out[base] = excl
    excl += v0
    if base + 1 < n:
        out[base + 1] = excl
    excl += v1
    if base + 2 < n:
        out[base + 2] = excl
    excl += v2
    if base + 3 < n:
        out[base + 3] = excl


TYPES = (int32[:], int32[:], int32[:], int64)
_jit = cuda.jit(kernel)


def run(values, out, totals, n):
    if n > 0:
        _jit[(n + 1023) // 1024, 256](values, out, totals, n)
