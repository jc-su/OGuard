import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
WARP = 32
WARPS = THREADS // WARP
BLOCKS_PER_SM = 8

TYPES = (types.float64[:], types.int8[:], types.float64[:], types.int64)


def kernel(values: DeviceNDArray, mask: DeviceNDArray, result: DeviceNDArray, n: int) -> None:
    # Grid-stride accumulation per thread, a warp-shuffle and shared-memory
    # reduction per block, and one atomic add per block.
    warp_sums = cuda.shared.array(WARPS, types.float64)
    tid = cuda.threadIdx.x
    lane = tid % WARP
    warp = tid // WARP
    i = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + tid
    stride = np.int64(cuda.gridDim.x) * cuda.blockDim.x
    acc = 0.0
    while i < n:
        if mask[i] != 0:
            acc += values[i]
        i += stride
    delta = WARP // 2
    while delta > 0:
        acc += cuda.shfl_down_sync(0xFFFFFFFF, acc, delta)
        delta //= 2
    if lane == 0:
        warp_sums[warp] = acc
    cuda.syncthreads()
    if warp == 0:
        acc = warp_sums[lane] if lane < WARPS else 0.0
        delta = WARPS // 2
        while delta > 0:
            acc += cuda.shfl_down_sync(0xFFFFFFFF, acc, delta)
            delta //= 2
        if lane == 0:
            cuda.atomic.add(result, 0, acc)


_kernel = cuda.jit(kernel)


def run(values: DeviceNDArray, mask: DeviceNDArray, result: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM)
    _kernel[blocks, THREADS](values, mask, result, n)
