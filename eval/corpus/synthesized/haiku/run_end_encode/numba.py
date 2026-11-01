from numba import cuda, types

def kernel(values, run_ends, run_values, count, n):
    i = cuda.grid(1)
    if i == 0 and n > 0:
        run_count = 0
        j = 0
        while j < n:
            current_val = values[j]
            run_start = j
            while j < n and values[j] == current_val:
                j += 1
            
            run_ends[run_count] = j
            run_values[run_count] = current_val
            run_count += 1
        
        count[0] = run_count

TYPES = (types.int64[:], types.int32[:], types.int64[:], types.int32[:], types.int64)

def run(values, run_ends, run_values, count, n):
    cuda.jit(kernel)[1, 1](values, run_ends, run_values, count, n)
