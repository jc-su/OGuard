import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256

TYPES = (types.int32[:], types.uint8[:], types.uint8[:], types.int64)


def kernel(offsets: DeviceNDArray, chars: DeviceNDArray, out: DeviceNDArray, n: int) -> None:
    # One thread per byte. The thread finds the string that holds its byte by a
    # binary search for the first offset above the byte position, then writes
    # the byte to the mirrored position of that string.
    t = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    first = np.int64(offsets[0])
    p = first + t
    if p < offsets[n]:
        lo = np.int64(1)
        hi = n
        while lo < hi:
            mid = (lo + hi) >> 1
            if offsets[mid] > p:
                hi = mid
            else:
                lo = mid + 1
        start = np.int64(offsets[lo - 1])
        end = np.int64(offsets[lo])
        out[start + end - 1 - p] = chars[p]


_kernel = cuda.jit(kernel)


def run(offsets: DeviceNDArray, chars: DeviceNDArray, out: DeviceNDArray, n: int) -> None:
    # chars holds offsets[n] bytes, which bounds the offsets[n] - offsets[0]
    # bytes that the strings cover.
    total = chars.shape[0]
    if total == 0:
        return
    blocks = (total + THREADS - 1) // THREADS
    _kernel[blocks, THREADS](offsets, chars, out, n)
