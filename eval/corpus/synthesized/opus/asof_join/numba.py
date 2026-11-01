"""asof_join: out[i] = largest j with right[j] <= left[i] and left[i] - right[j] <= tolerance,
or -1.

The largest j with right[j] <= left[i] is upper_bound(right, left[i]) - 1. Any
smaller j has right[j] at most as large, so it is farther from left[i]: if the
largest j is out of tolerance, there is no match. Values are in [-2^61, 2^61], so
left[i] - right[j] does not overflow.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32

TYPES = (
    types.int64[:],  # left
    types.int64[:],  # right
    types.int32[:],  # out
    types.int64,  # n
    types.int64,  # r
    types.int64,  # tolerance
)


def kernel(left, right, out, n, r, tolerance):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        x = left[i]
        lo = types.int64(0)
        hi = r
        while lo < hi:
            mid = lo + ((hi - lo) >> 1)
            if right[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        if j >= 0 and x - right[j] <= tolerance:
            out[i] = j
        else:
            out[i] = -1
        i += stride


_kernel = cuda.jit(kernel)


def run(left, right, out, n, r, tolerance):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](left, right, out, n, r, tolerance)
