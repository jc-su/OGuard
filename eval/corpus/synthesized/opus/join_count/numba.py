"""Join count: for each probe key, count the equal keys in the sorted build column."""

from numba import cuda, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int64[:], int64[:], int32[:], int64, int64)

THREADS = 256


def kernel(build, probe, counts, b, p):
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    while i < p:
        key = probe[i]
        # Lower bound: first position whose key is not less than key.
        lo = 0
        hi = b
        while lo < hi:
            mid = (lo + hi) >> 1
            if build[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        first = lo
        # Upper bound: first position whose key is greater than key.
        hi = b
        while lo < hi:
            mid = (lo + hi) >> 1
            if build[mid] <= key:
                lo = mid + 1
            else:
                hi = mid
        counts[i] = lo - first
        i += stride


_KERNEL = cuda.jit(kernel)


def run(
    build: DeviceNDArray,
    probe: DeviceNDArray,
    counts: DeviceNDArray,
    b: int,
    p: int,
) -> None:
    if p == 0:
        return
    blocks = (p + THREADS - 1) // THREADS
    _KERNEL[blocks, THREADS](build, probe, counts, b, p)
