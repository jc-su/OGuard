from numba import cuda, types
import numpy as np

TYPES = (
    types.int64[:],  # values
    types.int64[:],  # out
    types.int64,     # n
)


def kernel(values, out, n):
    idx = cuda.grid(1)
    if idx < n:
        if idx == 0:
            out[idx] = values[idx]
        else:
            out[idx] = values[idx] - values[idx - 1]


def run(values: np.ndarray, out: np.ndarray, n: int) -> None:
    if n == 0:
        return
    
    d_values = cuda.to_device(values)
    d_out = cuda.to_device(out)
    
    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock
    
    cuda.jit(kernel)[blockspergrid, threadsperblock](d_values, d_out, n)
    
    d_out.copy_to_host(out)
