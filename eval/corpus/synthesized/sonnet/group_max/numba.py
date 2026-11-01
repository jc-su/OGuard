import numpy as np
from numba import cuda, int32, int64

BLOCK = 256
SHARED_GROUPS = 1024
INT64_MIN = np.int64(np.iinfo(np.int64).min)


def kernel(group, value, best, n, capacity):
    # A small capacity with enough rows per block is reduced in a per-block copy in shared memory,
    # which is merged into best once per block; otherwise rows update best directly.
    local = cuda.shared.array(SHARED_GROUPS, int64)
    tid = cuda.threadIdx.x
    start = int64(cuda.blockIdx.x) * cuda.blockDim.x + tid
    stride = int64(cuda.gridDim.x) * cuda.blockDim.x
    if capacity <= SHARED_GROUPS and n >= 4 * capacity * cuda.gridDim.x:
        for k in range(tid, capacity, cuda.blockDim.x):
            local[k] = INT64_MIN
        cuda.syncthreads()
        for i in range(start, n, stride):
            cuda.atomic.max(local, group[i], value[i])
        cuda.syncthreads()
        for k in range(tid, capacity, cuda.blockDim.x):
            v = local[k]
            if v != INT64_MIN:
                cuda.atomic.max(best, k, v)
    else:
        for i in range(start, n, stride):
            cuda.atomic.max(best, group[i], value[i])


TYPES = (int32[:], int64[:], int64[:], int64, int64)


def run(group, value, best, n, capacity):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min(-(-n // BLOCK), 4 * sms))
    cuda.jit(kernel)[blocks, BLOCK](group, value, best, n, capacity)
