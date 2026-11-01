from numba import cuda, types
import numpy as np

TYPES = (types.float64[:], types.int32[:], types.float64[:], types.int64, types.int64)

def kernel(values, first, out, n, k):
    """Window function lag: out[i] = values[i - k] if i - k >= first[i], else NaN."""
    i = cuda.grid(1)
    if i < n:
        lag_idx = i - k
        first_row = first[i]

        if lag_idx >= first_row:
            out[i] = values[lag_idx]
        else:
            out[i] = float('nan')

def run(values, first, n, k):
    """Launch the lag kernel on device arrays."""
    values_dev = cuda.to_device(values)
    first_dev = cuda.to_device(first)
    out_dev = cuda.device_array(n, dtype=np.float64)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](values_dev, first_dev, out_dev, n, k)

    return out_dev.copy_to_host()
