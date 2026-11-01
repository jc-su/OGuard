from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.float64[:], types.int64, types.float64[:])

def kernel(offsets, values, g, sums):
    j = cuda.grid(1)
    if j < g:
        start = offsets[j]
        end = offsets[j + 1]
        total = 0.0
        for i in range(start, end):
            total += values[i]
        sums[j] = total

def run(offsets: np.ndarray, values: np.ndarray, sums: np.ndarray):
    g = offsets.size - 1
    threads_per_block = 256
    blocks = (g + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](offsets, values, np.int64(g), sums)
