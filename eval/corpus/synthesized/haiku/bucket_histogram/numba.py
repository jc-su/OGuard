from numba import cuda, types
import numpy as np

TYPES = (
    types.float64[:],  # values
    types.float64[:],  # bounds
    types.int32[:],    # counts
    types.int64,       # n
    types.int32,       # b
)


def kernel(values, bounds, counts, n, b):
    idx = cuda.grid(1)
    if idx < n:
        val = values[idx]
        # Count how many bounds <= val
        count = 0
        for i in range(b):
            if bounds[i] <= val:
                count += 1
        cuda.atomic.add(counts, count, 1)


def run(values: np.ndarray, bounds: np.ndarray, counts: np.ndarray, n: int, b: int) -> None:
    if n == 0:
        return

    counts[:] = 0

    d_values = cuda.to_device(values)
    d_bounds = cuda.to_device(bounds)
    d_counts = cuda.to_device(counts)

    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock

    cuda.jit(kernel)[blockspergrid, threadsperblock](d_values, d_bounds, d_counts, n, b)

    d_counts.copy_to_host(counts)
