"""bucketize: out[i] is the number of bounds that are <= values[i].

bounds is sorted ascending, so the count is the upper bound of values[i] in
bounds, which each thread finds by binary search.
"""

from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256


def kernel(
    values: DeviceNDArray,
    bounds: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
    b: int,
) -> None:
    i = cuda.grid(1)
    if i < n:
        v = values[i]
        lo = 0
        hi = b
        while lo < hi:
            mid = (lo + hi) >> 1
            if bounds[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo


TYPES = (
    types.float64[:],
    types.float64[:],
    types.int32[:],
    types.int64,
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    values: DeviceNDArray,
    bounds: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
    b: int,
) -> None:
    blocks = (n + THREADS - 1) // THREADS
    if blocks > 0:
        _kernel[blocks, THREADS](values, bounds, out, n, b)
