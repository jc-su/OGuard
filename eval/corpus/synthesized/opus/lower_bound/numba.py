import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256

TYPES = (types.int64[:], types.int64[:], types.int32[:], types.int64, types.int64)


def kernel(keys: DeviceNDArray, sorted: DeviceNDArray, out: DeviceNDArray, n: int, s: int) -> None:
    i = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i < n:
        key = keys[i]
        lo = np.int64(0)
        hi = s
        while lo < hi:
            mid = (lo + hi) >> 1
            if sorted[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo


_kernel = cuda.jit(kernel)


def run(keys: DeviceNDArray, sorted: DeviceNDArray, out: DeviceNDArray, n: int, s: int) -> None:
    if n == 0:
        return
    blocks = (n + THREADS - 1) // THREADS
    _kernel[blocks, THREADS](keys, sorted, out, n, s)
