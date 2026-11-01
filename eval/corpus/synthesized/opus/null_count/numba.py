"""Null count: add the number of rows whose validity bit is 0 to count[0]."""

from numba import cuda, int64, uint8
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (uint8[:], int64[:], int64)

THREADS = 256
BLOCKS_PER_SM = 4
FULL_MASK = 0xFFFFFFFF


def kernel(valid, count, n):
    nbytes = (n + 7) // 8
    stride = cuda.gridsize(1)
    j = cuda.grid(1)
    nulls = 0
    while j < nbytes:
        rows = n - j * 8
        if rows > 8:
            rows = 8
        nulls += rows - cuda.popc(valid[j] & ((1 << rows) - 1))
        j += stride

    # Sum over the warp, then over the warps of the block, then one atomic.
    offset = 16
    while offset > 0:
        nulls += cuda.shfl_down_sync(FULL_MASK, nulls, offset)
        offset //= 2
    partial = cuda.shared.array(32, int64)
    lane = cuda.laneid
    warp = cuda.threadIdx.x // 32
    if lane == 0:
        partial[warp] = nulls
    cuda.syncthreads()
    if warp == 0:
        nulls = 0
        if lane < cuda.blockDim.x // 32:
            nulls = partial[lane]
        offset = 16
        while offset > 0:
            nulls += cuda.shfl_down_sync(FULL_MASK, nulls, offset)
            offset //= 2
        if lane == 0 and nulls != 0:
            cuda.atomic.add(count, 0, nulls)


_KERNEL = cuda.jit(kernel)


def run(valid: DeviceNDArray, count: DeviceNDArray, n: int) -> None:
    nbytes = (n + 7) // 8
    if nbytes == 0:
        return
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = min((nbytes + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM)
    _KERNEL[blocks, THREADS](valid, count, n)
