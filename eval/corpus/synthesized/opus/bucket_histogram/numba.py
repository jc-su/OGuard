import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
BLOCKS_PER_SM = 8
# Counts per block are kept in shared memory when the b + 1 buckets fit in
# this many int32 slots (16 KiB); otherwise every row updates counts directly.
SHARED_BINS = 4096

TYPES = (types.float64[:], types.float64[:], types.int32[:], types.int64, types.int64)


def kernel(
    values: DeviceNDArray,
    bounds: DeviceNDArray,
    counts: DeviceNDArray,
    n: int,
    b: int,
) -> None:
    local = cuda.shared.array(SHARED_BINS, types.int32)
    tid = cuda.threadIdx.x
    buckets = b + 1
    in_shared = buckets <= SHARED_BINS
    if in_shared:
        k = tid
        while k < buckets:
            local[k] = 0
            k += THREADS
        cuda.syncthreads()
    i = np.int64(cuda.blockIdx.x) * cuda.blockDim.x + tid
    stride = np.int64(cuda.gridDim.x) * cuda.blockDim.x
    while i < n:
        v = values[i]
        lo = np.int64(0)
        hi = b
        while lo < hi:
            mid = (lo + hi) >> 1
            if bounds[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        if in_shared:
            cuda.atomic.add(local, lo, 1)
        else:
            cuda.atomic.add(counts, lo, 1)
        i += stride
    if in_shared:
        cuda.syncthreads()
        k = tid
        while k < buckets:
            c = local[k]
            if c != 0:
                cuda.atomic.add(counts, k, c)
            k += THREADS


_kernel = cuda.jit(kernel)


def run(
    values: DeviceNDArray,
    bounds: DeviceNDArray,
    counts: DeviceNDArray,
    n: int,
    b: int,
) -> None:
    if n == 0:
        return
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM)
    _kernel[blocks, THREADS](values, bounds, counts, n, b)
