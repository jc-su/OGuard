import numpy as np
from numba import cuda
from numba.types import int32, int64


def kernel(group, value, best, n, g):
    i = np.int64(cuda.blockIdx.x) * 256 + np.int64(cuda.threadIdx.x)
    if i < n:
        row = np.int32(i)
        v = value[i]
        k = group[i]
        # best[k] holds the row of the current maximum of group k, so value supplies its value.
        # A stale read is safe: stored rows only improve, and a failed swap returns the current row.
        cur = best[k]
        while True:
            if cur >= 0:
                cv = value[cur]
                if cv > v or (cv == v and cur < row):
                    break
            old = cuda.atomic.cas(best, k, cur, row)
            if old == cur:
                break
            cur = old


TYPES = (int32[:], int64[:], int32[:], int64, int64)

_kernel = cuda.jit(kernel)


def run(group, value, best, n, g):
    grid = max(1, (n + 255) // 256)
    _kernel[grid, 256](group, value, best, n, g)
