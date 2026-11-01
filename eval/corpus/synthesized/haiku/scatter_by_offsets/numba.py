from numba import cuda, types
import numpy as np

TYPES = (types.int8[:], types.int32[:], types.int64, types.int64, types.int32[:])

def kernel(flags, offsets, n, total, out):
    i = cuda.grid(1)
    if i < n:
        if flags[i] != 0:
            out[offsets[i]] = i

def run(flags: np.ndarray, offsets: np.ndarray, total: int, out: np.ndarray):
    n = flags.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](flags, offsets, np.int64(n), np.int64(total), out)
