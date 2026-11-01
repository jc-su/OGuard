"""join_materialize: for probe row i whose key matches build rows f .. f + c - 1,
out_build[offsets[i] + j] = f + j and out_probe[offsets[i] + j] = i.

The work is split by output position, so a probe row with many matches does not
serialize one thread. Each block takes tiles of 256 * ITEMS output positions; two
threads find the first and last probe row of the tile by binary search over
offsets, and each thread then finds the probe row of its first position within
that range and writes ITEMS contiguous positions, moving to the next nonempty
probe row when it reaches the end of one. f is the lower bound of probe[i] in the
sorted build column, found once per probe row a thread touches.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 16
ITEMS = 8

TYPES = (
    types.int64[:],  # build
    types.int64[:],  # probe
    types.int32[:],  # offsets
    types.int32[:],  # out_build
    types.int32[:],  # out_probe
    types.int64,  # b
    types.int64,  # p
)


@cuda.jit(device=True, inline=True)
def _upper_bound(a, lo, hi, x):
    """First index k in [lo, hi) with a[k] > x, or hi."""
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if types.int64(a[mid]) <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cuda.jit(device=True, inline=True)
def _lower_bound(a, n, x):
    """First index k in [0, n) with a[k] >= x, or n."""
    lo = types.int64(0)
    hi = n
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def kernel(build, probe, offsets, out_build, out_probe, b, p):
    tile_rows = cuda.shared.array(2, types.int64)
    tid = cuda.threadIdx.x
    total = types.int64(offsets[p])
    tile = types.int64(cuda.blockDim.x * ITEMS)
    t0 = types.int64(cuda.blockIdx.x) * tile
    while t0 < total:
        t1 = min(t0 + tile, total)
        if tid < 2:
            target = t0 if tid == 0 else t1 - 1
            tile_rows[tid] = _upper_bound(offsets, types.int64(0), p + 1, target) - 1
        cuda.syncthreads()
        row_lo = tile_rows[0]
        row_hi = tile_rows[1]
        cuda.syncthreads()

        q = t0 + tid * ITEMS
        q_end = min(q + ITEMS, t1)
        if q < q_end:
            r = _upper_bound(offsets, row_lo, row_hi + 1, q) - 1
            r_end = types.int64(offsets[r + 1])
            f = _lower_bound(build, b, probe[r]) + (q - offsets[r])
            while q < q_end:
                if q >= r_end:
                    while q >= r_end:
                        r += 1
                        r_end = types.int64(offsets[r + 1])
                    f = _lower_bound(build, b, probe[r])
                out_build[q] = f
                out_probe[q] = r
                f += 1
                q += 1
        t0 += types.int64(cuda.gridDim.x) * tile


_kernel = cuda.jit(kernel)


def run(build, probe, offsets, out_build, out_probe, b, p):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    _kernel[sms * BLOCKS_PER_SM, THREADS](build, probe, offsets, out_build, out_probe, b, p)
