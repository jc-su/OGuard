from numba import cuda, types
import numpy as np

TYPES = (types.float64[:], types.int64, types.float64, types.int32[:], types.int32[:])

def kernel(quantity, n, threshold, sel, count):
    i = cuda.grid(1)
    if i < n:
        if quantity[i] < threshold:
            idx = cuda.atomic.add(count, 0, 1)
            if idx < n:
                sel[idx] = types.int32(i)

def run(quantity: np.ndarray, threshold: float, sel: np.ndarray, count: np.ndarray):
    """Launch filter_select kernel."""
    n = quantity.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](quantity, np.int64(n), threshold, sel, count)
