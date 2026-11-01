from numba import cuda, types
import numpy as np

TYPES = (types.int32[:], types.int32[:], types.int32[:], types.int64)

def kernel(values, out, totals, n):
    """Tile scan: prefix sum within tiles of 1024 elements, and tile totals."""
    idx = cuda.grid(1)
    TILE_SIZE = 1024

    if idx < n:
        tile_idx = idx // TILE_SIZE
        local_idx = idx % TILE_SIZE
        tile_start = tile_idx * TILE_SIZE
        tile_end = min(tile_start + TILE_SIZE, n)

        # Compute prefix sum within the tile
        prefix_sum = 0
        for i in range(tile_start, tile_start + local_idx):
            prefix_sum += values[i]
        out[idx] = prefix_sum

        # Compute and store tile total (by first thread of tile)
        if local_idx == 0:
            total = 0
            for i in range(tile_start, tile_end):
                total += values[i]
            totals[tile_idx] = total

def run(values, n):
    """Launch the tile_scan kernel on device arrays."""
    values_dev = cuda.to_device(values)
    out_dev = cuda.device_array(n, dtype=np.int32)

    num_tiles = (n + 1023) // 1024
    totals_dev = cuda.device_array(num_tiles, dtype=np.int32)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](values_dev, out_dev, totals_dev, n)

    return out_dev.copy_to_host(), totals_dev.copy_to_host()
