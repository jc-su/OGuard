"""gather_nullable: out[i] = col[idx[i]], and NaN where idx[i] is -1."""

import math

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256


def kernel(
    col: DeviceNDArray,
    idx: DeviceNDArray,
    out: DeviceNDArray,
    m: int,
    n: int,
) -> None:
    i = cuda.grid(1)
    if i < m:
        j = idx[i]
        if j >= 0:
            out[i] = col[j]
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
    col: DeviceNDArray,
    idx: DeviceNDArray,
    out: DeviceNDArray,
    m: int,
    n: int,
) -> None:
    blocks = (m + THREADS - 1) // THREADS
    if blocks > 0:
        _kernel[blocks, THREADS](col, idx, out, m, n)
