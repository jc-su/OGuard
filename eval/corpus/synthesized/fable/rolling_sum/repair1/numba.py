"""rolling_sum: trailing window sum of width `window` (1 to 1024).

Each block produces 256 outputs. It stages its tile plus the window - 1
preceding values (at most 1279 doubles) in shared memory, computes an
inclusive prefix sum over them, and forms every output as the difference of
two prefix values, out[i] = P[i] - P[i - window]. That is two shared memory
reads per output instead of `window` reads. The prefix sums are local to
the block (at most 1279 terms), so the rounding error is of the same order
as that of summing the window directly.

Revision: the prefix sum is one block scan of five consecutive values per
thread instead of five strided scan passes, so no loop indexes memory: the
staging, the scan and the write-back are one statement per value, and the
cross-warp step is a shuffle scan. The staged length is capped at the
buffer size (a no-op, as window <= 1024) and the output index is tested
against the staged range. The tests always pass under the task's contract;
they state the bound of each access for the memory checker, which does not
carry loop invariants.
"""

from numba import cuda, float64, int64

TYPES = (float64[:], float64[:], int64, int64)
THREADS = 256
MAX_WINDOW = 1024
SMEM = THREADS + MAX_WINDOW  # the tile plus the halo of the widest window
ITEMS = SMEM // THREADS  # consecutive staged values per thread in the scan


@cuda.jit(device=True, inline="always")
def stage(s, values, h, e, l):
    if l < e:
        s[l] = values[h + l]


def kernel(values, out, n, window):
    s = cuda.shared.array(SMEM, float64)
    warp_sum = cuda.shared.array(8, float64)
    warp_offset = cuda.shared.array(8, float64)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5
    t0 = cuda.blockIdx.x * THREADS
    if t0 >= n:
        return
    tile = min(THREADS, n - t0)
    h = max(0, t0 - window + 1)  # first value the block needs
    e = min(t0 + tile - h, SMEM)  # number of values to stage

    # s[0:e] = values[h:h + e], coalesced.
    stage(s, values, h, e, tid)
    stage(s, values, h, e, tid + THREADS)
    stage(s, values, h, e, tid + 2 * THREADS)
    stage(s, values, h, e, tid + 3 * THREADS)
    stage(s, values, h, e, tid + 4 * THREADS)
    cuda.syncthreads()

    # Inclusive prefix sum of s[0:e]: each thread scans ITEMS consecutive
    # values in registers, the block scans the thread totals.
    b = tid * ITEMS
    x0 = s[b] if b < e else 0.0
    x1 = s[b + 1] if b + 1 < e else 0.0
    x2 = s[b + 2] if b + 2 < e else 0.0
    x3 = s[b + 3] if b + 3 < e else 0.0
    x4 = s[b + 4] if b + 4 < e else 0.0
    p0 = x0
    p1 = p0 + x1
    p2 = p1 + x2
    p3 = p2 + x3
    p4 = p3 + x4

    x = p4
    off = 1
    while off < 32:
        y = cuda.shfl_up_sync(0xFFFFFFFF, x, off)
        if lane >= off:
            x += y
        off <<= 1
    if lane == 31:
        warp_sum[wid] = x
    cuda.syncthreads()
    if tid < 32:
        c = warp_sum[tid] if tid < 8 else 0.0
        z = c
        off = 1
        while off < 32:
            y = cuda.shfl_up_sync(0xFFFFFFFF, z, off)
            if lane >= off:
                z += y
            off <<= 1
        if tid < 8:
            warp_offset[tid] = z - c
    cuda.syncthreads()

    prefix = warp_offset[wid] + x - p4  # sum of s[0:b]
    if b < e:
        s[b] = prefix + p0
    if b + 1 < e:
        s[b + 1] = prefix + p1
    if b + 2 < e:
        s[b + 2] = prefix + p2
    if b + 3 < e:
        s[b + 3] = prefix + p3
    if b + 4 < e:
        s[b + 4] = prefix + p4
    cuda.syncthreads()

    i = t0 + tid
    if i < n:
        l = i - h
        if l >= 0 and l < e:
            v = s[l]
            lw = l - window
            if lw >= 0 and lw < e:
                v -= s[lw]
            out[i] = v


_kernel = cuda.jit(kernel)


def run(values, out, n, window):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](values, out, n, window)
