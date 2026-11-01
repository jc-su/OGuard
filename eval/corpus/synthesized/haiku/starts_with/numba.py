from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.uint8[:], types.uint8[:], types.int64, types.int8[:])

def kernel(offsets, chars, prefix, k, match):
    i = cuda.grid(1)
    if i < offsets.size - 1:
        start = offsets[i]
        end = offsets[i + 1]
        length = end - start

        if length < k:
            match[i] = 0
        else:
            matches = True
            for j in range(k):
                if chars[start + j] != prefix[j]:
                    matches = False
                    break
            match[i] = 1 if matches else 0

def run(offsets: np.ndarray, chars: np.ndarray, prefix: np.ndarray, match: np.ndarray):
    n = offsets.size - 1
    k = prefix.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](offsets, chars, prefix, np.int64(k), match)
