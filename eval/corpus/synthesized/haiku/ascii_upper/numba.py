from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.uint8[:], types.uint8[:], types.int64)

def kernel(offsets, chars, out, n):
    """Convert ASCII lowercase to uppercase in-place for all strings."""
    idx = cuda.grid(1)
    # Total number of characters to process
    total_chars = offsets[n]

    if idx < total_chars:
        c = chars[idx]
        # Convert lowercase a-z to uppercase A-Z
        if 97 <= c <= 122:  # 97 = 'a', 122 = 'z'
            out[idx] = c - 32
        else:
            out[idx] = c

def run(offsets, chars, n):
    """Launch the ascii_upper kernel on device arrays."""
    offsets_dev = cuda.to_device(offsets)
    chars_dev = cuda.to_device(chars)
    total_chars = offsets[n]
    out_dev = cuda.device_array(total_chars, dtype=np.uint8)

    # Launch kernel
    threads_per_block = 256
    blocks = (total_chars + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](offsets_dev, chars_dev, out_dev, n)

    return out_dev.copy_to_host()
