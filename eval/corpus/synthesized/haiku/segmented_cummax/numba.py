from numba import cuda, types

def kernel(values, first, out, n):
    i = cuda.grid(1)
    if i < n:
        segment_start = first[i]
        max_val = values[segment_start]
        
        for j in range(segment_start, i + 1):
            if values[j] > max_val:
                max_val = values[j]
        
        out[i] = max_val

TYPES = (types.int64[:], types.int32[:], types.int64[:], types.int64)

def run(values, first, out, n):
    cuda.jit(kernel)[(n + 255) // 256, 256](values, first, out, n)
