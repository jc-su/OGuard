from numba import cuda, types
import numpy as np

TYPES = (types.float64[:], types.uint8[:], types.float64[:], types.float64[:], types.int64)

def kernel(a, valid, b, out, n):
    """Coalesce: out[i] = a[i] if row i is valid, else b[i]."""
    idx = cuda.grid(1)
    if idx < n:
        # Check if row idx is valid: bit (idx % 8) of byte (idx // 8)
        byte_idx = idx // 8
        bit_idx = idx % 8
        is_valid = (valid[byte_idx] >> bit_idx) & 1
        out[idx] = a[idx] if is_valid else b[idx]

def run(a, b, n):
    """Launch the coalesce kernel on device arrays."""
    # Compute valid bitmap size
    valid_size = (n + 7) // 8
    valid = np.zeros(valid_size, dtype=np.uint8)
    # Mark all rows as valid (set all bits to 1)
    for i in range(valid_size):
        valid[i] = 0xFF if i * 8 + 8 <= n else (0xFF >> (8 - (n % 8)))

    # Convert to device arrays
    a_dev = cuda.to_device(a)
    valid_dev = cuda.to_device(valid)
    b_dev = cuda.to_device(b)
    out_dev = cuda.device_array(n, dtype=np.float64)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](a_dev, valid_dev, b_dev, out_dev, n)

    return out_dev.copy_to_host()
