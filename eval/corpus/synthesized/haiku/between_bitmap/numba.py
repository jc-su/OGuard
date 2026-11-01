from numba import cuda, types
import numpy as np

def kernel(values, valid, out, n, lo, hi):
    i = cuda.grid(1)
    if i < n:
        byte_idx = i // 8
        bit_idx = i % 8
        valid_bit = (valid[byte_idx] >> bit_idx) & 1

        if valid_bit and lo <= values[i] <= hi:
            out[byte_idx] = out[byte_idx] | (1 << bit_idx)
        else:
            out[byte_idx] = out[byte_idx] & ~(1 << bit_idx)

TYPES = (types.int64[:], types.uint8[:], types.uint8[:], types.int64, types.int64, types.int64)

def run(values, valid, out, n, lo, hi):
    out[:] = 0
    cuda.jit(kernel)[(n + 255) // 256, 256](values, valid, out, n, lo, hi)
