"""coalesce: out[i] = a[i] where row i of a is valid, and b[i] otherwise."""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256


def kernel(
    a: DeviceNDArray,
    valid: DeviceNDArray,
    b: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    i = cuda.grid(1)
    if i < n:
        if (valid[i >> 3] >> (i & 7)) & 1:
            out[i] = a[i]
        else:
            out[i] = b[i]


TYPES = (
    types.float64[:],
    types.uint8[:],
    types.float64[:],
    types.float64[:],
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    a: DeviceNDArray,
    valid: DeviceNDArray,
    b: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    blocks = (n + THREADS - 1) // THREADS
    if blocks > 0:
        _kernel[blocks, THREADS](a, valid, b, out, n)
