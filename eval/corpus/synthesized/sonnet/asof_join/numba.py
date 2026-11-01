import numpy as np
from numba import cuda
from numba.types import int32, int64


def kernel(left, right, out, n, r, tolerance):
    i = np.int64(cuda.blockIdx.x) * 256 + cuda.threadIdx.x
    if i < n:
        x = left[i]
        # Upper bound: the first row of right that is greater than x.
        lo = np.int64(0)
        hi = r
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if right[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        # The largest row with right[j] <= x is the closest to x, so if it is
        # outside the tolerance every smaller row is too.
        j = lo - 1
        if j >= 0 and x - right[j] <= tolerance:
            out[i] = j
        else:
            out[i] = -1


TYPES = (int64[:], int64[:], int32[:], int64, int64, int64)

_kernel = cuda.jit(kernel)


def run(left, right, out, n, r, tolerance):
    grid = max(1, (n + 255) // 256)
    _kernel[grid, 256](left, right, out, n, r, tolerance)
