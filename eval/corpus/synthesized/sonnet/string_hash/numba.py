import numpy as np
from numba import cuda, int32, int64, uint8, uint32


def kernel(offsets, chars, out, n):
    i = int64(cuda.blockIdx.x) * 256 + int64(cuda.threadIdx.x)
    if i >= n:
        return
    h = np.uint64(2166136261)
    prime = np.uint64(16777619)
    mask = np.uint64(0xFFFFFFFF)
    for p in range(offsets[i], offsets[i + 1]):
        h = ((h ^ np.uint64(chars[p])) * prime) & mask
    out[i] = np.uint32(h)


TYPES = (int32[:], uint8[:], uint32[:], int64)

_kernel = cuda.jit(kernel)


def run(offsets, chars, out, n):
    grid = max(1, (n + 255) // 256)
    _kernel[grid, 256](offsets, chars, out, n)
