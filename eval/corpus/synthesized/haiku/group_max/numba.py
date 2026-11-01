from numba import cuda, types
import numpy as np

TYPES = (
    types.int32[:],  # group
    types.int64[:],  # value
    types.int64[:],  # best
    types.int64,     # n
    types.int32,     # capacity
)


def kernel(group, value, best, n, capacity):
    idx = cuda.grid(1)
    if idx < n:
        g = group[idx]
        v = value[idx]
        if 0 <= g < capacity:
            # Atomic max
            cuda.atomic.max(best, g, v)


def run(group: np.ndarray, value: np.ndarray, best: np.ndarray, n: int, capacity: int) -> None:
    if n == 0:
        return

    # Initialize best with smallest int64
    best[:] = -(2**63)

    d_group = cuda.to_device(group)
    d_value = cuda.to_device(value)
    d_best = cuda.to_device(best)

    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock

    cuda.jit(kernel)[blockspergrid, threadsperblock](d_group, d_value, d_best, n, capacity)

    d_best.copy_to_host(best)
