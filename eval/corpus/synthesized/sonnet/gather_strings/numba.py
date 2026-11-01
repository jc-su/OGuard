import numpy as np
from numba import cuda, int32, int64, uint8

THREADS = 256
WARPS = THREADS // 32


def kernel(offsets, chars, idx, out_offsets, out_chars, n, m):
    # One warp per output string; the lanes copy its bytes with a stride of 32.
    tid = cuda.threadIdx.x
    lane = tid & 31
    i = np.int64(cuda.blockIdx.x) * WARPS + (tid >> 5)
    if i < m:
        src = np.int64(idx[i])
        start = np.int64(offsets[src])
        length = np.int64(offsets[src + 1]) - start
        dst = np.int64(out_offsets[i])
        k = np.int64(lane)
        while k < length:
            out_chars[dst + k] = chars[start + k]
            k += 32


TYPES = (int32[:], uint8[:], int32[:], int32[:], uint8[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(offsets, chars, idx, out_offsets, out_chars, n, m):
    grid = max(1, (m + WARPS - 1) // WARPS)
    _kernel[grid, THREADS](offsets, chars, idx, out_offsets, out_chars, n, m)
