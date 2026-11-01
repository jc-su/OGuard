from numba import cuda, types

def kernel(ts, values, out, n, range_val):
    i = cuda.grid(1)
    if i < n:
        ts_i = ts[i]
        lower_bound = ts_i - range_val
        
        total = 0
        for j in range(n):
            if ts[j] >= lower_bound and ts[j] <= ts_i:
                total += values[j]
        
        out[i] = total

TYPES = (types.int64[:], types.int64[:], types.int64[:], types.int64, types.int64)

def run(ts, values, out, n, range_val):
    cuda.jit(kernel)[(n + 255) // 256, 256](ts, values, out, n, range_val)
