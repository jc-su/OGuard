from numba import cuda, types

def kernel(ts, out, n, unit):
    i = cuda.grid(1)
    if i < n:
        ts_val = ts[i]
        if ts_val >= 0:
            out[i] = (ts_val // unit) * unit
        else:
            if ts_val % unit == 0:
                out[i] = ts_val
            else:
                out[i] = ((ts_val // unit) - 1) * unit

TYPES = (types.int64[:], types.int64[:], types.int64, types.int64)

def run(ts, out, n, unit):
    cuda.jit(kernel)[(n + 255) // 256, 256](ts, out, n, unit)
