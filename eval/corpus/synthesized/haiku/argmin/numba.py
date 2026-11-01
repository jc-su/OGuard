from numba import cuda, types

def kernel(values, row, n):
    i = cuda.grid(1)
    if i == 0 and n > 0:
        min_val = values[0]
        min_row = 0
        
        for j in range(1, n):
            if values[j] < min_val:
                min_val = values[j]
                min_row = j
        
        row[0] = min_row

TYPES = (types.int64[:], types.int32[:], types.int64)

def run(values, row, n):
    cuda.jit(kernel)[1, 1](values, row, n)
