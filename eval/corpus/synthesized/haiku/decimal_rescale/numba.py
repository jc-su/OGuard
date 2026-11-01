from numba import cuda, types
import numpy as np

def kernel(values, out, overflow, n, shift):
    i = cuda.grid(1)
    if i < n:
        v = values[i]
        if shift >= 0:
            power = 1
            for _ in range(shift):
                power *= 10
            
            result = v * power
            min_int64 = -(2**63)
            max_int64 = 2**63 - 1
            
            if result < min_int64 or result > max_int64:
                overflow[i] = 1
                out[i] = 0
            else:
                overflow[i] = 0
                out[i] = result
        else:
            shift_abs = -shift
            power = 1
            for _ in range(shift_abs):
                power *= 10
            
            abs_v = abs(v)
            rounded = (abs_v + power // 2) // power
            if v < 0:
                out[i] = -rounded
            else:
                out[i] = rounded
            overflow[i] = 0

TYPES = (types.int64[:], types.int64[:], types.int8[:], types.int64, types.int64)

def run(values, out, overflow, n, shift):
    cuda.jit(kernel)[(n + 255) // 256, 256](values, out, overflow, n, shift)
