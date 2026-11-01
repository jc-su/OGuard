from numba import cuda, types
import numpy as np

TYPES = (
    types.int32[:],  # offsets
    types.int64[:],  # values
    types.int64[:],  # maxes
    types.int32,     # g
)


def kernel(offsets, values, maxes, g):
    idx = cuda.grid(1)
    if idx < g:
        start = offsets[idx]
        end = offsets[idx + 1]

        # Initialize with smallest int64
        max_val = -(2**63)

        # Find max in this segment
        for i in range(start, end):
            if values[i] > max_val:
                max_val = values[i]

        maxes[idx] = max_val


def run(offsets: np.ndarray, values: np.ndarray, maxes: np.ndarray, g: int) -> None:
    if g == 0:
        return

    d_offsets = cuda.to_device(offsets)
    d_values = cuda.to_device(values)
    d_maxes = cuda.to_device(maxes)

    threadsperblock = 256
    blockspergrid = (g + threadsperblock - 1) // threadsperblock

    cuda.jit(kernel)[blockspergrid, threadsperblock](d_offsets, d_values, d_maxes, g)

    d_maxes.copy_to_host(maxes)
