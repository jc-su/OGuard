from numba import cuda, types

def kernel(values, out, n, window):
    """Compute rolling sum."""
    idx = cuda.grid(1)
    if idx < n:
        start = max(0, idx - window + 1)
        total = 0.0
        for i in range(start, idx + 1):
            total += values[i]
        out[idx] = total

TYPES = (types.float64[:], types.float64[:], types.int64, types.int64)

def run(values_np, out_np, n, window):
    """Launch kernel on device arrays."""
    values = cuda.to_device(values_np)
    out = cuda.to_device(out_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](values, out, n, window)

    return out.copy_to_host()
