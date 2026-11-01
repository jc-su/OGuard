from numba import cuda, types
import math

def kernel(values, result, n):
    """Compute min and max of values using block-wise reduction."""
    idx = cuda.grid(1)
    block_size = cuda.blockDim.x
    tid = cuda.threadIdx.x
    
    shared_min = cuda.shared.array(512, dtype=types.float64)
    shared_max = cuda.shared.array(512, dtype=types.float64)
    
    local_min = float('inf')
    local_max = float('-inf')
    
    if idx < n:
        val = values[idx]
        if val < local_min:
            local_min = val
        if val > local_max:
            local_max = val
    
    shared_min[tid] = local_min
    shared_max[tid] = local_max
    cuda.syncthreads()
    
    s = block_size // 2
    while s > 0:
        if tid < s:
            if shared_min[tid + s] < shared_min[tid]:
                shared_min[tid] = shared_min[tid + s]
            if shared_max[tid + s] > shared_max[tid]:
                shared_max[tid] = shared_max[tid + s]
        cuda.syncthreads()
        s //= 2
    
    if tid == 0:
        cuda.atomic.min(result, 0, shared_min[0])
        cuda.atomic.max(result, 1, shared_max[0])

TYPES = (types.float64[:], types.float64[:], types.int64)

def run(values_np, result_np, n):
    """Launch kernel on device arrays."""
    values = cuda.to_device(values_np)
    result = cuda.to_device(result_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](values, result, n)

    return result.copy_to_host()
