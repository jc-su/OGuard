from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.uint8[:], types.int32[:], types.uint8[:], types.int8[:], types.int64)

def kernel(a_offsets, a_chars, b_offsets, b_chars, match, n):
    """Compare strings: match[i] = 1 if strings are equal, 0 otherwise."""
    i = cuda.grid(1)
    if i < n:
        # Get string offsets and lengths
        a_start = a_offsets[i]
        a_end = a_offsets[i + 1]
        a_len = a_end - a_start

        b_start = b_offsets[i]
        b_end = b_offsets[i + 1]
        b_len = b_end - b_start

        # Check if lengths differ
        if a_len != b_len:
            match[i] = 0
        else:
            # Compare bytes
            is_equal = True
            for j in range(a_len):
                if a_chars[a_start + j] != b_chars[b_start + j]:
                    is_equal = False
                    break
            match[i] = 1 if is_equal else 0

def run(a_offsets, a_chars, b_offsets, b_chars, n):
    """Launch the string_equal kernel on device arrays."""
    a_offsets_dev = cuda.to_device(a_offsets)
    a_chars_dev = cuda.to_device(a_chars)
    b_offsets_dev = cuda.to_device(b_offsets)
    b_chars_dev = cuda.to_device(b_chars)
    match_dev = cuda.device_array(n, dtype=np.int8)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](a_offsets_dev, a_chars_dev, b_offsets_dev, b_chars_dev, match_dev, n)

    return match_dev.copy_to_host()
