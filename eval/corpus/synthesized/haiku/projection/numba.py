from numba import cuda, types
import numpy as np

TYPES = (types.float64[:], types.float64[:], types.int64, types.float64[:])

def kernel(price, discount, n, out):
    i = cuda.grid(1)
    if i < n:
        out[i] = price[i] * (1.0 - discount[i])

def run(price: np.ndarray, discount: np.ndarray, out: np.ndarray):
    n = price.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](price, discount, np.int64(n), out)
