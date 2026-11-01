import numpy as np
from numba import cuda
from numba.types import int32, int64


def kernel(keys, table_keys, table_rows, n, capacity):
    i = np.int64(cuda.blockIdx.x) * 256 + np.int64(cuda.threadIdx.x)
    if i < n:
        key = keys[i]
        mask = capacity - 1
        h = (np.uint64(key) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)
        slot = np.int64(h) & mask
        # Slots only change from the empty marker -1 to a key, so every thread that
        # probes the same key reaches the same slot.
        while True:
            old = cuda.atomic.cas(table_keys, slot, np.int64(-1), key)
            if old == -1 or old == key:
                cuda.atomic.min(table_rows, slot, np.int32(i))
                break
            slot = (slot + 1) & mask


TYPES = (int64[:], int64[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(keys, table_keys, table_rows, n, capacity):
    grid = max(1, (n + 255) // 256)
    _kernel[grid, 256](keys, table_keys, table_rows, n, capacity)
