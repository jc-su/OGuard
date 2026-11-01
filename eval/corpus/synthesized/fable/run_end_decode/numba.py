"""run_end_decode: expand an Arrow run-end encoded column.

One thread per output row. Row i belongs to the first run j with
run_ends[j] > i (an upper bound search). Each block first finds the run
range of its 256 rows with two searches over all of run_ends (one in warp 0,
one in warp 1), then every thread searches only inside that range, which is
a few runs wide when runs are long.
"""

from numba import cuda, float64, int32, int64

TYPES = (float64[:], int32[:], float64[:], int64, int64)
THREADS = 256


@cuda.jit(device=True)
def upper_bound(run_ends, lo, hi, row):
    while lo < hi:
        mid = (lo + hi) >> 1
        if run_ends[mid] <= row:
            lo = mid + 1
        else:
            hi = mid
    return lo


def kernel(run_values, run_ends, out, r, n):
    bounds = cuda.shared.array(2, int64)
    tid = cuda.threadIdx.x
    t0 = cuda.blockIdx.x * THREADS
    if t0 >= n:
        return
    last = min(t0 + THREADS, n) - 1
    if tid == 0:
        bounds[0] = upper_bound(run_ends, 0, r, t0)
    elif tid == 32:
        bounds[1] = upper_bound(run_ends, 0, r, last) + 1
    cuda.syncthreads()
    i = t0 + tid
    if i <= last:
        j = upper_bound(run_ends, bounds[0], bounds[1], i)
        out[i] = run_values[j]


_kernel = cuda.jit(kernel)


def run(run_values, run_ends, out, r, n):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](run_values, run_ends, out, r, n)
