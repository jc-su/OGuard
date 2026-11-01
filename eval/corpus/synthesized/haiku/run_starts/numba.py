from numba import cuda, types
import numpy as np

TYPES = (
    types.int64[:],  # values
    types.int8[:],   # flags
    types.int64,     # n
)


def kernel(values, flags, n):
    idx = cuda.grid(1)
    if idx < n:
        if idx == 0:
            flags[idx] = 1
        else:
            flags[idx] = 1 if values[idx] != values[idx - 1] else 0


def run(values: np.ndarray, flags: np.ndarray, n: int) -> None:
    if n == 0:
        return

    d_values = cuda.to_device(values)
    d_flags = cuda.to_device(flags)

    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock

    cuda.jit(kernel)[blockspergrid, threadsperblock](d_values, d_flags, n)

    d_flags.copy_to_host(flags)
