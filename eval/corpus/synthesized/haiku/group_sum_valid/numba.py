from numba import cuda, types
import numpy as np

def kernel(group, value, valid, counts, sums, n, g):
    i = cuda.grid(1)
    if i < n:
        byte_idx = i // 8
        bit_idx = i % 8
        valid_bit = (valid[byte_idx] >> bit_idx) & 1

        if valid_bit:
            grp = group[i]
            cuda.atomic.add(counts, grp, 1)
            cuda.atomic.add(sums, grp, value[i])

TYPES = (types.int32[:], types.int64[:], types.uint8[:], types.int64[:], types.int64[:], types.int64, types.int64)

def run(group, value, valid, counts, sums, n, g):
    cuda.jit(kernel)[(n + 255) // 256, 256](group, value, valid, counts, sums, n, g)
