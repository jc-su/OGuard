from numba import cuda
import numpy as np


def kernel(keys, part, counts, n, p):
    """Partition keys by hash value."""
    idx = cuda.grid(1)
    if idx < n:
        key = keys[idx]
        # Hash: (x * 0x9E3779B97F4A7C15) >> 32
        # Need to handle as unsigned 64-bit
        h = ((key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
        partition = h % p
        part[idx] = int(partition)
        # Increment count for this partition
        cuda.atomic.add(counts, int(partition), 1)


TYPES = (
    cuda.int64[:],
    cuda.int32[:],
    cuda.int32[:],
    cuda.int64,
    cuda.int64,
)


def run(keys, part, counts, n, p, grid_size=None):
    """Launch the hash_partition kernel."""
    if grid_size is None:
        grid_size = max(1, (n + 255) // 256)
    kernel_jit = cuda.jit(kernel)
    kernel_jit[(grid_size, 1), (256, 1)](keys, part, counts, n, p)
