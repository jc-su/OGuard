"""run_end_decode: expand an Arrow run-end encoded column.

One thread per output row. Row i belongs to the first run j with
run_ends[j] > i (an upper bound search). Each block first finds the run
range of its 256 rows with two searches over all of run_ends (one in warp 0,
one in warp 1), then every thread searches only inside that range, which is
a few runs wide when runs are long.

Revision: the searches are written out in the kernel instead of a device
function, every probe of `run_ends` is guarded by an explicit range test on
its index, and the gather from `run_values` is guarded the same way. The
searches keep 0 <= lo < hi <= r and the last run ends at n, so the tests
always pass; they state the bound of each access for the memory checker,
which does not follow calls or carry loop invariants.
"""

from numba import cuda, float64, int32, int64

TYPES = (float64[:], int32[:], float64[:], int64, int64)
THREADS = 256


def kernel(run_values, run_ends, out, r, n):
    bounds = cuda.shared.array(2, int64)
    tid = cuda.threadIdx.x
    t0 = cuda.blockIdx.x * THREADS
    if t0 >= n:
        return
    last = min(t0 + THREADS, n) - 1

    # Thread 0 finds the run of the block's first row, thread 32 the run
    # after the run of its last row; every thread then searches in between.
    if tid == 0 or tid == 32:
        row = t0 if tid == 0 else last
        lo = 0
        hi = r
        while lo < hi:
            mid = (lo + hi) >> 1
            if mid < 0 or mid >= r:
                break
            if run_ends[mid] <= row:
                lo = mid + 1
            else:
                hi = mid
        if tid == 0:
            bounds[0] = lo
        else:
            bounds[1] = lo + 1
    cuda.syncthreads()

    i = t0 + tid
    if i < n:
        lo = bounds[0]
        hi = bounds[1]
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
