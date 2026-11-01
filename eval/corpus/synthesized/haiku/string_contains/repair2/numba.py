from numba import cuda, types

def kernel(offsets, chars, pattern, match, n, k):
    """Check if pattern is contained in each string."""
    idx = cuda.grid(1)
    if idx < n:
        # Calculate offsets safely
        # offsets has n+1 elements
        # Ensure idx + 1 is within the offsets array bounds
        if idx < n:
            start = offsets[idx]
            # Bounds check: idx + 1 must be < len(offsets) which is n+1
            if idx + 1 < n + 1:
                end = offsets[idx + 1]
            else:
                end = start
            
            str_len = end - start
            
            if k == 0:
                match[idx] = 1
                return
            
            if str_len < k:
                match[idx] = 0
                return
            
            found = 0
            for i in range(str_len - k + 1):
                all_match = 1
                for j in range(k):
                    pos = start + i + j
                    # Explicit bounds check before accessing chars and pattern
                    if pos >= 0 and pos < len(chars) and j >= 0 and j < k:
                        if chars[pos] != pattern[j]:
                            all_match = 0
                            break
                    else:
                        all_match = 0
                        break
                if all_match:
                    found = 1
                    break
            
            match[idx] = found

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
