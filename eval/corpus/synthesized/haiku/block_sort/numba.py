from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.int32[:], types.int64)

def kernel(keys, out, n):
    """Sort each tile of 1024 elements."""
    idx = cuda.grid(1)
    TILE_SIZE = 1024

    if idx < n:
        # Copy input to output
        out[idx] = keys[idx]

    cuda.syncthreads()

    # Sort within each tile using simple approach (bubble sort for demonstration)
    # For production, use bitonic sort or other efficient sort
    tile_idx = idx // TILE_SIZE
    local_idx = idx % TILE_SIZE
    tile_start = tile_idx * TILE_SIZE
    tile_end = min(tile_start + TILE_SIZE, n)
    tile_size = tile_end - tile_start

    if idx < n:
        # Simple bubble sort within tile
        for i in range(tile_size):
            for j in range(tile_size - i - 1):
                idx1 = tile_start + j
                idx2 = tile_start + j + 1
                if out[idx1] > out[idx2]:
                    # Swap
                    temp = out[idx1]
                    out[idx1] = out[idx2]
                    out[idx2] = temp
            cuda.syncthreads()

def run(keys, n):
    """Launch the block_sort kernel on device arrays."""
    keys_dev = cuda.to_device(keys)
    out_dev = cuda.device_array(n, dtype=np.int32)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](keys_dev, out_dev, n)

    return out_dev.copy_to_host()
