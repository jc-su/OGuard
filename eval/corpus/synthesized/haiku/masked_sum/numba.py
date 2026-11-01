from numba import cuda, types
import numpy as np

TYPES = (
    types.float64[:],  # values
    types.int8[:],     # mask
    types.float64[:],  # result
    types.int64,       # n
)


def kernel(values, mask, result, n):
    idx = cuda.grid(1)
    if idx < n:
        if mask[idx] != 0:
            cuda.atomic.add(result, 0, values[idx])


def run(values: np.ndarray, mask: np.ndarray, result: np.ndarray, n: int) -> None:
    if n == 0:
        result[0] = 0.0
        return

    result[0] = 0.0

    d_values = cuda.to_device(values)
    d_mask = cuda.to_device(mask)
    d_result = cuda.to_device(result)

    threadsperblock = 256
    blockspergrid = (n + threadsperblock - 1) // threadsperblock

    cuda.jit(kernel)[blockspergrid, threadsperblock](d_values, d_mask, d_result, n)

    d_result.copy_to_host(result)
