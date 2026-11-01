from numba import cuda, types
import numpy as np

def kernel(offsets, chars, out, n):
    i = cuda.grid(1)
    if i < n:
        start = offsets[i]
        end = offsets[i + 1]
        
        h = 2166136261
        for j in range(start, end):
            c = chars[j]
            h = ((h ^ c) * 16777619) & 0xFFFFFFFF
        
        out[i] = h

TYPES = (types.int32[:], types.uint8[:], types.uint32[:], types.int64)

def run(offsets, chars, out, n):
    cuda.jit(kernel)[(n + 255) // 256, 256](offsets, chars, out, n)
