import numpy as np
from numba import cuda, int64, uint8


def kernel(source, out, n, offset):
    j = int64(cuda.blockIdx.x) * 256 + int64(cuda.threadIdx.x)
    nbytes = (n + 7) // 8
    if j >= nbytes:
        return
    s = offset >> 3
    b = offset & 7
    source_bytes = (offset + n + 7) // 8
    value = np.int64(source[s + j]) >> b
    if b != 0 and s + j + 1 < source_bytes:
        value |= np.int64(source[s + j + 1]) << (8 - b)
    value &= 0xFF
    remaining = n - 8 * j
    if remaining < 8:
        value &= (np.int64(1) << remaining) - 1
    out[j] = np.uint8(value)


TYPES = (uint8[:], uint8[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(source, out, n, offset):
    grid = max(1, ((n + 7) // 8 + 255) // 256)
    _kernel[grid, 256](source, out, n, offset)
