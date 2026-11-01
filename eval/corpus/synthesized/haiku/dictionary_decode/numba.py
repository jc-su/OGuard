from numba import cuda, types

def kernel(codes, dictionary, out, n, d):
    """Decode dictionary-encoded column."""
    idx = cuda.grid(1)
    if idx < n:
        code = codes[idx]
        out[idx] = dictionary[code]

TYPES = (types.int32[:], types.float64[:], types.float64[:], types.int64, types.int64)

def run(codes_np, dictionary_np, out_np, n, d):
    """Launch kernel on device arrays."""
    codes = cuda.to_device(codes_np)
    dictionary = cuda.to_device(dictionary_np)
    out = cuda.to_device(out_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](codes, dictionary, out, n, d)

    return out.copy_to_host()
