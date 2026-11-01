from numba import cuda, types
import numpy as np

TYPES = (types.int8[:], types.int32[:], types.int64, types.int64, types.int32[:])

def kernel(flags, offsets, n, total, out):
    i = cuda.grid(1)
    if i < n:
        if flags[i] != 0:
            offset = offsets[i]
            if offset >= 0 and offset < total:
                out[offset] = i

def run(flags: np.ndarray, offsets: np.ndarray, total: int, out: np.ndarray):
    n = flags.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    if blocks == 0:
        blocks = 1
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](flags, offsets, np.int64(n), np.int64(total), out)
