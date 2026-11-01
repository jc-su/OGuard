from numba import cuda, types
import numpy as np

TYPES = (types.float64[:], types.int32[:], types.int64, types.int64, types.float64[:])

def kernel(col, sel, m, n, out):
    i = cuda.grid(1)
    if i < m:
        out[i] = col[sel[i]]

def run(col: np.ndarray, sel: np.ndarray, out: np.ndarray):
    """Launch gather kernel."""
    m = sel.size
    n = col.size
    threads_per_block = 256
    blocks = (m + threads_per_block - 1) // threads_per_block
    kernel_jitted = cuda.jit(kernel)
    kernel_jitted[blocks, threads_per_block](col, sel, np.int64(m), np.int64(n), out)
