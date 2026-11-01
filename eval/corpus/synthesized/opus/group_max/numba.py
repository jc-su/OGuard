import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256

TYPES = (types.int32[:], types.int64[:], types.int64[:], types.int64, types.int64)


def kernel(
    group: DeviceNDArray,
    value: DeviceNDArray,
    best: DeviceNDArray,
    n: int,
    capacity: int,
) -> None:
    i = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i < n:
        k = group[i]
        v = value[i]
        # best[k] only grows, so a value not above a (possibly stale) read of
        # best[k] cannot change it and the atomic can be skipped.
        if v > best[k]:
            cuda.atomic.max(best, k, v)


_kernel = cuda.jit(kernel)


def run(
    group: DeviceNDArray,
    value: DeviceNDArray,
    best: DeviceNDArray,
    n: int,
    capacity: int,
) -> None:
    if n == 0:
        return
    blocks = (n + THREADS - 1) // THREADS
    _kernel[blocks, THREADS](group, value, best, n, capacity)
