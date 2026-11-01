import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
WARP = 32
INT64_MIN = np.iinfo(np.int64).min

TYPES = (types.int32[:], types.int64[:], types.int64[:], types.int64)


def kernel(offsets: DeviceNDArray, values: DeviceNDArray, maxes: DeviceNDArray, g: int) -> None:
    # One warp per segment. The condition seg < g is uniform within a warp
    # because the block size is a multiple of the warp size, so every lane
    # takes part in the shuffles.
    t = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + cuda.threadIdx.x
    seg = t // WARP
    lane = t % WARP
    if seg < g:
        start = np.int64(offsets[seg])
        end = np.int64(offsets[seg + 1])
        best = INT64_MIN
        j = start + lane
        while j < end:
            v = values[j]
            if v > best:
                best = v
            j += WARP
        delta = WARP // 2
        while delta > 0:
            other = cuda.shfl_down_sync(0xFFFFFFFF, best, delta)
            if other > best:
                best = other
            delta //= 2
        if lane == 0:
            maxes[seg] = best


_kernel = cuda.jit(kernel)


def run(offsets: DeviceNDArray, values: DeviceNDArray, maxes: DeviceNDArray, g: int) -> None:
    if g == 0:
        return
    warps_per_block = THREADS // WARP
    blocks = (g + warps_per_block - 1) // warps_per_block
    _kernel[blocks, THREADS](offsets, values, maxes, g)
