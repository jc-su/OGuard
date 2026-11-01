import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256

TYPES = (types.int32[:], types.float64[:], types.float64[:], types.int64, types.int64)


def kernel(idx: DeviceNDArray, values: DeviceNDArray, out: DeviceNDArray, m: int, n: int) -> None:
    i = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i < m:
        cuda.atomic.add(out, idx[i], values[i])


_kernel = cuda.jit(kernel)


def run(idx: DeviceNDArray, values: DeviceNDArray, out: DeviceNDArray, m: int, n: int) -> None:
    if m == 0:
        return
    blocks = (m + THREADS - 1) // THREADS
    _kernel[blocks, THREADS](idx, values, out, m, n)
