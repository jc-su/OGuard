import numpy as np
from numba import cuda
from numba.types import int64, uint8


def kernel(values, valid, out, n, lo, hi):
    i = np.int64(cuda.blockIdx.x) * 256 + cuda.threadIdx.x
    keep = False
    if i < n:
        bit = (valid[i >> 3] >> (i & 7)) & 1
        x = values[i]
        keep = bit != 0 and x >= lo and x <= hi
    # One ballot covers 32 consecutive rows, which are 4 bytes of the output.
    # Every thread takes part, including those beyond n, whose predicate is 0.
    word = cuda.ballot_sync(0xFFFFFFFF, keep)
    lane = cuda.threadIdx.x & 31
    nbytes = (n + 7) >> 3
    if lane < 4:
        byte = (np.int64(cuda.blockIdx.x) * 256 + (cuda.threadIdx.x - lane)) // 8 + lane
        if byte < nbytes:
            out[byte] = np.uint8((word >> (8 * lane)) & 0xFF)


TYPES = (int64[:], uint8[:], uint8[:], int64, int64, int64)

_kernel = cuda.jit(kernel)


def run(values, valid, out, n, lo, hi):
    grid = max(1, (n + 255) // 256)
    _kernel[grid, 256](values, valid, out, n, lo, hi)
