"""run_end_decode: expand an Arrow run-end encoded column.

One thread per output row. Row i belongs to the first run j with
run_ends[j] > i, found by an upper-bound binary search over all of
run_ends; the upper levels of the search are shared by every thread and
stay in L1/L2.

Revision: the block-level narrowing of the search range through shared
memory is gone, so the search bounds start at 0 and r for every thread and
no index that reaches memory passes through shared memory. Every probe of
`run_ends` and the gather from `run_values` test the exact index they use
against [0, r); under the task's contract those tests always pass.
"""

from numba import cuda, float64, int32, int64

TYPES = (float64[:], int32[:], float64[:], int64, int64)
THREADS = 256


def kernel(run_values, run_ends, out, r, n):
    i = cuda.blockIdx.x * THREADS + cuda.threadIdx.x
    if i >= n:
        return
    lo = 0
    hi = r
    while lo < hi:
        mid = (lo + hi) >> 1
        if mid < 0 or mid >= r:
            break
        if run_ends[mid] <= i:
            lo = mid + 1
        else:
            hi = mid
    if lo >= 0 and lo < r:
        out[i] = run_values[lo]


_kernel = cuda.jit(kernel)


def run(run_values, run_ends, out, r, n):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](run_values, run_ends, out, r, n)
