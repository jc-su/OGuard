import numpy as np
from numba import cuda, int64, uint32

THREADS = 256


def kernel(packed, out, n, w, bits):
    # One thread per code. A code spans two words when its bits cross a word boundary.
    i = np.int64(cuda.blockIdx.x) * 256 + cuda.threadIdx.x
    if i < n:
        p = i * bits
        word = p >> 5
        shift = p & 31
        v = np.uint64(packed[word])
        if shift + bits > 32:
            v |= np.uint64(packed[word + 1]) << np.uint64(32)
        v >>= np.uint64(shift)
        mask = (np.uint64(1) << np.uint64(bits)) - np.uint64(1)
        out[i] = np.uint32(v & mask)


TYPES = (uint32[:], uint32[:], int64, int64, int64)

_kernel = cuda.jit(kernel)


def run(packed, out, n, w, bits):
    grid = max(1, (n + THREADS - 1) // THREADS)
    _kernel[grid, THREADS](packed, out, n, w, bits)
