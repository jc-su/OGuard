import numpy as np
from numba import cuda, float64, int8, int32, int64, uint8, uint32


H1 = np.uint64(0x9E3779B97F4A7C15)
H2 = np.uint64(0xC2B2AE3D27D4EB4F)
S32 = np.uint64(32)


def kernel(keys, bits, n, words):
    i = cuda.grid(1)
    if i < n:
        k = np.uint64(keys[i])
        mask = np.uint64(32 * words - 1)
        p1 = np.int64(((k * H1) >> S32) & mask)
        p2 = np.int64(((k * H2) >> S32) & mask)
        cuda.atomic.or_(bits, p1 >> 5, np.uint32(1 << (p1 & 31)))
        cuda.atomic.or_(bits, p2 >> 5, np.uint32(1 << (p2 & 31)))


TYPES = (int64[:], uint32[:], int64, int64)
_jit = cuda.jit(kernel)


def run(keys, bits, n, words):
    if n > 0:
        _jit[(n + 255) // 256, 256](keys, bits, n, words)
