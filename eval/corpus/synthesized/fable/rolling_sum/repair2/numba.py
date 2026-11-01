"""rolling_sum: trailing window sum of width `window` (1 to 1024).

Each block produces 256 outputs. It stages the tile and a fixed halo of
MAX_WINDOW preceding rows (TILE = 1280 doubles) in shared memory behind
MAX_WINDOW zeros, rows before the column or before every window of the tile
as zeros. For staged position u it forms z[u] = s[u] - s[u - window]; the
inclusive scan of z at u is exactly the sum of the window ending at u, and
every partial sum of the scan is itself a window sum, so no large prefix
sums are subtracted from each other. The scan runs on 5 consecutive values
per thread in registers, a warp shuffle scan of the thread totals and a
shuffle scan of the warp totals.

Revision: the halo is fixed instead of window-dependent, so no index is
clamped with min or max; every row index is the block's first staged row
plus a thread-affine offset, tested against [0, n) right before the load;
and the zeros in front of the staged rows make every shared index affine
in the thread id and the window with no range test at all (u - window is
at least u - MAX_WINDOW >= 0). The thread id is masked to the block size
and shared indices are unsigned, so no index carries a sign fix-up.
"""

from numba import cuda, float64, int64, uint64

TYPES = (float64[:], float64[:], int64, int64)
THREADS = 256
MAX_WINDOW = 1024
TILE = THREADS + MAX_WINDOW  # 1280 staged rows: the halo of the widest window and the tile
SMEM = MAX_WINDOW + TILE  # MAX_WINDOW zeros, then the staged rows
ITEMS = TILE // THREADS  # 5 consecutive positions per thread in the scan


@cuda.jit(device=True, inline="always")
def stage(s, values, n, base, first, l):
    """s[MAX_WINDOW + l] = values[base + l], or 0 when that row is missing or before every window of the tile."""
    v = 0.0
    if l >= first:
        g = base + l
        if g >= 0 and g < n:
            v = values[g]
    s[MAX_WINDOW + l] = v


@cuda.jit(device=True, inline="always")
def warp_scan(x, lane):
    """Inclusive scan of x over the warp."""
    off = 1
    while off < 32:
        y = cuda.shfl_up_sync(0xFFFFFFFF, x, off)
        if lane >= off:
            x += y
        off <<= 1
    return x


def kernel(values, out, n, window):
    s = cuda.shared.array(SMEM, float64)
    warp_sum = cuda.shared.array(32, float64)  # 8 used; one slot per lane of warp 0
    warp_offset = cuda.shared.array(32, float64)
    tid = cuda.threadIdx.x & (THREADS - 1)
    lane = tid & 31
    wid = tid >> 5
    t0 = cuda.blockIdx.x * THREADS
    if t0 >= n:
        return
    base = t0 - MAX_WINDOW  # row of s[MAX_WINDOW]
    first = MAX_WINDOW - window  # staged rows before this one lie outside every window of the tile

    s[tid] = 0.0
    s[tid + THREADS] = 0.0
    s[tid + 2 * THREADS] = 0.0
    s[tid + 3 * THREADS] = 0.0
    stage(s, values, n, base, first, tid)
    stage(s, values, n, base, first, tid + THREADS)
    stage(s, values, n, base, first, tid + 2 * THREADS)
    stage(s, values, n, base, first, tid + 3 * THREADS)
    stage(s, values, n, base, first, tid + 4 * THREADS)
    cuda.syncthreads()

    # z[u] = s[u] - s[u - window] for the thread's 5 consecutive positions, scanned in registers.
    u = MAX_WINDOW + tid * ITEMS
    z0 = s[u] - s[uint64(u - window)]
    z1 = s[u + 1] - s[uint64(u + 1 - window)]
    z2 = s[u + 2] - s[uint64(u + 2 - window)]
    z3 = s[u + 3] - s[uint64(u + 3 - window)]
    z4 = s[u + 4] - s[uint64(u + 4 - window)]
    p0 = z0
    p1 = p0 + z1
    p2 = p1 + z2
    p3 = p2 + z3
    p4 = p3 + z4

    x = warp_scan(p4, lane)
    if lane == 31:
        warp_sum[wid] = x
    cuda.syncthreads()
    if tid < 32:
        c = 0.0
        if tid < 8:
            c = warp_sum[tid]
        warp_offset[tid] = warp_scan(c, lane) - c
    cuda.syncthreads()

    prefix = warp_offset[wid] + x - p4  # scan of z before position u
    s[u] = prefix + p0
    s[u + 1] = prefix + p1
    s[u + 2] = prefix + p2
    s[u + 3] = prefix + p3
    s[u + 4] = prefix + p4
    cuda.syncthreads()

    i = t0 + tid
    if i < n:
        out[i] = s[2 * MAX_WINDOW + tid]


_kernel = cuda.jit(kernel)


def run(values, out, n, window):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](values, out, n, window)
