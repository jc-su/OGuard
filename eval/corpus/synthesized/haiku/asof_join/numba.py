from numba import cuda, types
import numpy as np

def kernel(left, right, out, n, r, tolerance):
    i = cuda.grid(1)
    if i < n:
        left_val = left[i]
        best_idx = -1

        for j in range(r):
            if right[j] <= left_val and left_val - right[j] <= tolerance:
                best_idx = j

        out[i] = best_idx

TYPES = (types.int64[:], types.int64[:], types.int32[:], types.int64, types.int64, types.int64)

def run(left, right, out, n, r, tolerance):
    cuda.jit(kernel)[(n + 255) // 256, 256](left, right, out, n, r, tolerance)
