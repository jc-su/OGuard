import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
TILE = 1024

TYPES = (types.int32[:], types.int32[:], types.int64)


def kernel(values: DeviceNDArray, out: DeviceNDArray, n: int) -> None:
    # Reads are contiguous per warp; the mirrored writes of a warp also cover
    # one contiguous range, so both sides stay coalesced.
    i = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    if i < n:
        start = (i // TILE) * TILE
        end = min(start + TILE, n)
        out[start + end - 1 - i] = values[i]


_kernel = cuda.jit(kernel)


def run(values: DeviceNDArray, out: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    blocks = (n + THREADS - 1) // THREADS
    _kernel[blocks, THREADS](values, out, n)
