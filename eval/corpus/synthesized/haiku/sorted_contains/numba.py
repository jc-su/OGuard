from numba import cuda, types

def kernel(keys, members, match, n, s):
    """Check if each key is in the sorted members set."""
    idx = cuda.grid(1)
    if idx < n:
        key = keys[idx]
        
        left, right = 0, s
        found = False
        
        while left < right:
            mid = (left + right) // 2
            if members[mid] == key:
                found = True
                break
            elif members[mid] < key:
                left = mid + 1
            else:
                right = mid
        
        match[idx] = 1 if found else 0

TYPES = (types.int64[:], types.int64[:], types.int8[:], types.int64, types.int64)

def run(keys_np, members_np, match_np, n, s):
    """Launch kernel on device arrays."""
    keys = cuda.to_device(keys_np)
    members = cuda.to_device(members_np)
    match = cuda.to_device(match_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](keys, members, match, n, s)

    return match.copy_to_host()
