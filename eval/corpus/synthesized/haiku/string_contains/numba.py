from numba import cuda, types

def kernel(offsets, chars, pattern, match, n, k):
    """Check if pattern is contained in each string."""
    idx = cuda.grid(1)
    if idx < n:
        start = offsets[idx]
        end = offsets[idx + 1]
        str_len = end - start
        
        if k == 0:
            match[idx] = 1
            return
        
        if str_len < k:
            match[idx] = 0
            return
        
        found = False
        for i in range(str_len - k + 1):
            all_match = True
            for j in range(k):
                if chars[start + i + j] != pattern[j]:
                    all_match = False
                    break
            if all_match:
                found = True
                break
        
        match[idx] = 1 if found else 0

TYPES = (types.int32[:], types.uint8[:], types.uint8[:], types.int8[:], types.int64, types.int64)

def run(offsets_np, chars_np, pattern_np, match_np, n, k):
    """Launch kernel on device arrays."""
    offsets = cuda.to_device(offsets_np)
    chars = cuda.to_device(chars_np)
    pattern = cuda.to_device(pattern_np)
    match = cuda.to_device(match_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](offsets, chars, pattern, match, n, k)

    return match.copy_to_host()
