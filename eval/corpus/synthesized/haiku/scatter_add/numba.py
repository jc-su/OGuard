from numba import cuda, types
import numpy as np

TYPES = (
    types.int32[:],    # idx
    types.float64[:],  # values
    types.float64[:],  # out
    types.int64,       # m
    types.int64,       # n
)


def kernel(idx, values, out, m, n):
    i = cuda.grid(1)
    if i < m:
        out_idx = idx[i]
        if out_idx >= 0 and out_idx < n:
            cuda.atomic.add(out, out_idx, values[i])


def run(idx: np.ndarray, values: np.ndarray, out: np.ndarray, m: int, n: int) -> None:
    if m == 0:
        return
    
    d_idx = cuda.to_device(idx)
    d_values = cuda.to_device(values)
    d_out = cuda.to_device(out)
    
    threadsperblock = 256
    blockspergrid = (m + threadsperblock - 1) // threadsperblock
    
    cuda.jit(kernel)[blockspergrid, threadsperblock](d_idx, d_values, d_out, m, n)
    
    d_out.copy_to_host(out)
