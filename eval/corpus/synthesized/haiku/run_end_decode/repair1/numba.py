from numba import cuda, types

def kernel(run_values, run_ends, out, r, n):
    """Decode Arrow run-end encoded column using binary search."""
    idx = cuda.grid(1)
    if idx < n:
        left, right = 0, r - 1
        while left < right:
            mid = (left + right) // 2
            if run_ends[mid] <= idx:
                left = mid + 1
            else:
                right = mid
        # Clamp left to valid range
        left = min(left, r - 1)
        if left >= 0 and left < r:
            out[idx] = run_values[left]

TYPES = (types.float64[:], types.int32[:], types.float64[:], types.int64, types.int64)

def run(run_values_np, run_ends_np, out_np, r, n):
    """Launch kernel on device arrays."""
    run_values = cuda.to_device(run_values_np)
    run_ends = cuda.to_device(run_ends_np)
    out = cuda.to_device(out_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](run_values, run_ends, out, r, n)

    return out.copy_to_host()
