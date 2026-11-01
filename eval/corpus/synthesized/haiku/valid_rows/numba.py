import numpy as np
from numba import cuda, types

def kernel(valid, rows, count, n):
    """Extract valid row IDs from Arrow validity bitmap."""
    idx = cuda.grid(1)
    if idx < n:
        byte_idx = idx // 8
        bit_idx = idx % 8
        if byte_idx < valid.size:
            is_valid = (valid[byte_idx] >> bit_idx) & 1
            if is_valid:
                pos = cuda.atomic.add(count, 0, 1)
                if pos < rows.size:
                    rows[pos] = idx

TYPES = (types.uint8[:], types.int32[:], types.int32[:], types.int64)

def run(valid_np, rows_np, count_np, n):
    """Launch kernel on device arrays."""
    valid = cuda.to_device(valid_np)
    rows = cuda.to_device(rows_np)
    count = cuda.to_device(count_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](valid, rows, count, n)

    return rows.copy_to_host(), count.copy_to_host()
