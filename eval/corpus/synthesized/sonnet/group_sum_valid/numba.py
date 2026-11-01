import numpy as np
from numba import cuda
from numba.types import int32, int64, uint8


def kernel(group, value, valid, counts, sums, n, g):
    # Per-block private tables; g is at most 1024.
    local_counts = cuda.shared.array(1024, int64)
    local_sums = cuda.shared.array(1024, int64)
    t = cuda.threadIdx.x
    for k in range(t, g, 256):
        local_counts[k] = 0
        local_sums[k] = 0
    cuda.syncthreads()
    start = np.int64(cuda.blockIdx.x) * 256 + t
    stride = np.int64(cuda.gridDim.x) * 256
    for i in range(start, n, stride):
        if (valid[i >> 3] >> (i & 7)) & 1:
            k = group[i]
            cuda.atomic.add(local_counts, k, 1)
            cuda.atomic.add(local_sums, k, value[i])
    cuda.syncthreads()
    for k in range(t, g, 256):
        if local_counts[k] != 0:
            cuda.atomic.add(counts, k, local_counts[k])
            cuda.atomic.add(sums, k, local_sums[k])


TYPES = (int32[:], int64[:], uint8[:], int64[:], int64[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(group, value, valid, counts, sums, n, g):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    grid = max(1, min((n + 255) // 256, 8 * sms))
    _kernel[grid, 256](group, value, valid, counts, sums, n, g)
