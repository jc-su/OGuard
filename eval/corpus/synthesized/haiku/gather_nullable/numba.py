from numba import cuda, types
import numpy as np

TYPES = (types.float64[:], types.int32[:], types.float64[:], types.int64, types.int64)

def kernel(col, idx, out, m, n):
    """Gather with nullable support: out[i] = col[idx[i]] or NaN if idx[i] == -1."""
    i = cuda.grid(1)
    if i < m:
        index = idx[i]
        if index == -1:
            out[i] = float('nan')
        else:
            out[i] = col[index]

def run(col, idx, m, n):
    """Launch the gather_nullable kernel on device arrays."""
    col_dev = cuda.to_device(col)
    idx_dev = cuda.to_device(idx)
    out_dev = cuda.device_array(m, dtype=np.float64)

    # Launch kernel
    threads_per_block = 256
    blocks = (m + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](col_dev, idx_dev, out_dev, m, n)

    return out_dev.copy_to_host()
