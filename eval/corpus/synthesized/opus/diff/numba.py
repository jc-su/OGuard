import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256

TYPES = (types.int64[:], types.int64[:], types.int64)


def kernel(values: DeviceNDArray, out: DeviceNDArray, n: int) -> None:
    i = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i < n:
        if i == 0:
            out[i] = values[i]
        else:
            out[i] = values[i] - values[i - 1]


_kernel = cuda.jit(kernel)


def run(values: DeviceNDArray, out: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    blocks = (n + THREADS - 1) // THREADS
    _kernel[blocks, THREADS](values, out, n)
