from numba import cuda, types

def kernel(a, b, out, m, p):
    """Merge two sorted arrays."""
    idx = cuda.grid(1)
    total = m + p

    if idx < total:
        i, j = 0, 0
        
        for k in range(idx):
            # Make bounds check immediately precede array access
            if i >= m:
                j += 1
            elif j >= p:
                i += 1
            elif i < m and j < p:
                # Bounds check is explicit here before array access
                if a[i] <= b[j]:
                    i += 1
                else:
                    j += 1
            # If none of the above, we've reached bounds and exit loop
        
        # Final selection with explicit bounds checks
        if i >= m and j < p:
            out[idx] = b[j]
        elif j >= p and i < m:
            out[idx] = a[i]
        elif i < m and j < p:
            # Both are in bounds, compare before accessing
            if a[i] <= b[j]:
                out[idx] = a[i]
            else:
                out[idx] = b[j]

TYPES = (types.int64[:], types.int64[:], types.int64[:], types.int64, types.int64)

def run(a_np, b_np, out_np, m, p):
    """Launch kernel on device arrays."""
    a = cuda.to_device(a_np)
    b = cuda.to_device(b_np)
    out = cuda.to_device(out_np)

    total = m + p
    threadsperblock = 256
    blockspergrid = max(1, (total + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](a, b, out, m, p)

    return out.copy_to_host()
