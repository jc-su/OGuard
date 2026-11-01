"""gather_strings: out_chars[out_offsets[i] : out_offsets[i + 1]] = bytes of string idx[i].

The work is split by output byte, so a long string does not serialize one thread.
Each block takes tiles of 256 * BYTES_PER_THREAD output bytes; two threads find
the first and last output row of the tile by binary search over out_offsets, and
each thread then finds the row of its first byte within that range and copies
BYTES_PER_THREAD contiguous bytes, moving to the next row (skipping empty rows)
when it reaches the end of one.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 16
BYTES_PER_THREAD = 16

TYPES = (
    types.int32[:],  # offsets
    types.uint8[:],  # chars
    types.int32[:],  # idx
    types.int32[:],  # out_offsets
    types.uint8[:],  # out_chars
    types.int64,  # n
    types.int64,  # m
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


def kernel(offsets, chars, idx, out_offsets, out_chars, n, m):
    tile_rows = cuda.shared.array(2, types.int64)
    tid = cuda.threadIdx.x
    total = types.int64(out_offsets[m])
    tile = types.int64(cuda.blockDim.x * BYTES_PER_THREAD)
    t0 = types.int64(cuda.blockIdx.x) * tile
    while t0 < total:
        t1 = min(t0 + tile, total)
        if tid < 2:
            target = t0 if tid == 0 else t1 - 1
            tile_rows[tid] = _upper_bound(out_offsets, types.int64(0), m + 1, target) - 1
        cuda.syncthreads()
        row_lo = tile_rows[0]
        row_hi = tile_rows[1]
        cuda.syncthreads()

        p = t0 + tid * BYTES_PER_THREAD
        p_end = min(p + BYTES_PER_THREAD, t1)
        if p < p_end:
            r = _upper_bound(out_offsets, row_lo, row_hi + 1, p) - 1
            r_end = types.int64(out_offsets[r + 1])
            src = types.int64(offsets[idx[r]]) + (p - out_offsets[r])
            while p < p_end:
                while p >= r_end:
                    r += 1
                    r_end = types.int64(out_offsets[r + 1])
                    src = types.int64(offsets[idx[r]])
                out_chars[p] = chars[src]
                src += 1
                p += 1
        t0 += types.int64(cuda.gridDim.x) * tile


_kernel = cuda.jit(kernel)


def run(offsets, chars, idx, out_offsets, out_chars, n, m):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    _kernel[sms * BLOCKS_PER_SM, THREADS](offsets, chars, idx, out_offsets, out_chars, n, m)
