from numba import cuda, types
import numpy as np

TYPES = (types.uint32[:], types.int64, types.int32[:])

def kernel(keys, shift, counts):
    i = cuda.grid(1)
    if i < keys.size:
        key = keys[i]
        byte_val = (key >> shift) & 255
        cuda.atomic.add(counts, byte_val, 1)

def run(keys: np.ndarray, shift: int, counts: np.ndarray):
    n = keys.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](keys, np.int64(shift), counts)
