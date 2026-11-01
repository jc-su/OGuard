from numba import cuda, types
import numpy as np

def kernel(offsets, chars, idx, out_offsets, out_chars, n, m):
    i = cuda.grid(1)
    if i < m:
        src_idx = idx[i]
        src_start = offsets[src_idx]
        src_end = offsets[src_idx + 1]
        dst_start = out_offsets[i]

        length = src_end - src_start
        for j in range(length):
            out_chars[dst_start + j] = chars[src_start + j]

TYPES = (types.int32[:], types.uint8[:], types.int32[:], types.int32[:], types.uint8[:], types.int64, types.int64)

def run(offsets, chars, idx, out_offsets, out_chars, n, m):
    cuda.jit(kernel)[(m + 255) // 256, 256](offsets, chars, idx, out_offsets, out_chars, n, m)
