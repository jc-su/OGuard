import numpy as np
from numba import cuda, int32, int64, uint8

THREADS = 256
WARPS = THREADS // 32


def kernel(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    # One warp per row; the lanes copy a's bytes, then b's bytes, with a stride of 32.
    tid = cuda.threadIdx.x
    lane = tid & 31
    i = np.int64(cuda.blockIdx.x) * WARPS + (tid >> 5)
    if i < n:
        a_start = np.int64(a_offsets[i])
        a_len = np.int64(a_offsets[i + 1]) - a_start
        b_start = np.int64(b_offsets[i])
        b_len = np.int64(b_offsets[i + 1]) - b_start
        dst = np.int64(out_offsets[i])
        k = np.int64(lane)
        while k < a_len:
            out_chars[dst + k] = a_chars[a_start + k]
            k += 32
        k = np.int64(lane)
        while k < b_len:
            out_chars[dst + a_len + k] = b_chars[b_start + k]
            k += 32


TYPES = (int32[:], uint8[:], int32[:], uint8[:], int32[:], uint8[:], int64)

_kernel = cuda.jit(kernel)


def run(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    grid = max(1, (n + WARPS - 1) // WARPS)
    _kernel[grid, THREADS](a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n)
