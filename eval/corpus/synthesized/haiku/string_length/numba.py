from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.uint8[:], types.int64[:])

def kernel(offsets, chars, lengths):
    i = cuda.grid(1)
    if i < offsets.size - 1:
        start = offsets[i]
        end = offsets[i + 1]
        count = 0
        pos = start
        while pos < end:
            byte = chars[pos]
            if byte < 128:
                count += 1
                pos += 1
            elif byte < 224:
                pos += 2
                count += 1
            elif byte < 240:
                pos += 3
                count += 1
            else:
                pos += 4
                count += 1
        lengths[i] = count

def run(offsets: np.ndarray, chars: np.ndarray, lengths: np.ndarray):
    n = offsets.size - 1
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](offsets, chars, lengths)
