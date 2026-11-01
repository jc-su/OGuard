from numba import cuda, types
import numpy as np

TYPES = (types.float64[:], types.float64[:], types.int32[:], types.int64, types.int64)

def kernel(values, bounds, out, n, b):
    """Bucketize: out[i] = number of bounds <= values[i]."""
    idx = cuda.grid(1)
    if idx < n:
        value = values[idx]
        # Binary search or linear search for simplicity
        count = 0
        for j in range(b):
            if bounds[j] <= value:
                count += 1
            else:
                break  # bounds are sorted, so can stop early
        out[idx] = count

def run(values, bounds, n, b):
    """Launch the bucketize kernel on device arrays."""
    values_dev = cuda.to_device(values)
    bounds_dev = cuda.to_device(bounds)
    out_dev = cuda.device_array(n, dtype=np.int32)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](values_dev, bounds_dev, out_dev, n, b)

    return out_dev.copy_to_host()
