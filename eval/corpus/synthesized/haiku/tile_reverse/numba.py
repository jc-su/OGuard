from numba import cuda, types
import numpy as np

TYPES = (
    types.int32[:],  # values
    types.int32[:],  # out
    types.int64,     # n
)

TILE_SIZE = 1024


def kernel(values, out, n):
    idx = cuda.grid(1)
    if idx < n:
        tile_id = idx // TILE_SIZE
        pos_in_tile = idx % TILE_SIZE
        tile_start = tile_id * TILE_SIZE
        tile_end = min(tile_start + TILE_SIZE, n)
        tile_len = tile_end - tile_start
        
        # Position in tile, reversed
        rev_pos = tile_len - 1 - pos_in_tile
        out[idx] = values[tile_start + rev_pos]


def run(values: np.ndarray, out: np.ndarray, n: int) -> None:
    if n == 0:
        return
    
    d_values = cuda.to_device(values)
    d_out = cuda.to_device(out)
    
    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock
    
    cuda.jit(kernel)[blockspergrid, threadsperblock](d_values, d_out, n)
    
    d_out.copy_to_host(out)
