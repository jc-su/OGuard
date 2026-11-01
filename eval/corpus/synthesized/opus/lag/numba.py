"""lag: out[i] = values[i - k] when row i - k is in row i's partition, else NaN."""

import math

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256


def kernel(
    values: DeviceNDArray,
    first: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
    k: int,
) -> None:
    i = cuda.grid(1)
    if i < n:
        j = i - k
        if j >= first[i]:
            out[i] = values[j]
        else:
            out[i] = math.nan


TYPES = (
    types.float64[:],
    types.int32[:],
    types.float64[:],
    types.int64,
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    values: DeviceNDArray,
    first: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
    k: int,
) -> None:
    blocks = (n + THREADS - 1) // THREADS
    if blocks > 0:
        _kernel[blocks, THREADS](values, first, out, n, k)
