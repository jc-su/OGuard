from numba import cuda, types

def kernel(keys, cursors, out, n, shift):
    """Scatter row ids by byte value (radix sort scatter)."""
    idx = cuda.grid(1)
    if idx < n:
        key = keys[idx]
        byte_val = (key >> shift) & 0xFF
        pos = cuda.atomic.add(cursors, int(byte_val), 1)
        # Ensure pos is within bounds before writing
        if pos >= 0 and pos < n:
            out[pos] = int(idx)

TYPES = (types.uint32[:], types.int32[:], types.int32[:], types.int64, types.int64)

def run(keys_np, cursors_np, out_np, n, shift):
    """Launch kernel on device arrays."""
    keys = cuda.to_device(keys_np)
    cursors = cuda.to_device(cursors_np)
    out = cuda.to_device(out_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](keys, cursors, out, n, shift)

    return out.copy_to_host(), cursors.copy_to_host()
