"""string_concat: out_chars[out_offsets[i] : out_offsets[i + 1]] = string i of a followed by
string i of b.

The work is split by output byte, so a long string does not serialize one thread.
Each block takes tiles of 256 * BYTES_PER_THREAD output bytes; two threads find the
first and last output row of the tile by binary search over out_offsets, and each
thread then finds the row of its first byte within that range and copies
BYTES_PER_THREAD contiguous bytes, taking byte k of row i from a when
k < length of a[i] and from b otherwise, and moving to the next row (skipping empty
rows) when it reaches the end of one.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 16
BYTES_PER_THREAD = 16

TYPES = (
    types.int32[:],  # a_offsets
    types.uint8[:],  # a_chars
    types.int32[:],  # b_offsets
    types.uint8[:],  # b_chars
    types.int32[:],  # out_offsets
    types.uint8[:],  # out_chars
    types.int64,  # n
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


def kernel(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    tile_rows = cuda.shared.array(2, types.int64)
    tid = cuda.threadIdx.x
    total = types.int64(out_offsets[n])
    tile = types.int64(cuda.blockDim.x * BYTES_PER_THREAD)
    t0 = types.int64(cuda.blockIdx.x) * tile
    while t0 < total:
        t1 = min(t0 + tile, total)
        if tid < 2:
            target = t0 if tid == 0 else t1 - 1
            tile_rows[tid] = _upper_bound(out_offsets, types.int64(0), n + 1, target) - 1
        cuda.syncthreads()
        row_lo = tile_rows[0]
        row_hi = tile_rows[1]
        cuda.syncthreads()

        p = t0 + tid * BYTES_PER_THREAD
        p_end = min(p + BYTES_PER_THREAD, t1)
        if p < p_end:
            r = _upper_bound(out_offsets, row_lo, row_hi + 1, p) - 1
            row_start = types.int64(out_offsets[r])
            row_end = types.int64(out_offsets[r + 1])
            a_begin = types.int64(a_offsets[r])
            a_len = types.int64(a_offsets[r + 1]) - a_begin
            b_begin = types.int64(b_offsets[r])
            while p < p_end:
                if p >= row_end:
                    while p >= row_end:
                        r += 1
                        row_end = types.int64(out_offsets[r + 1])
                    row_start = types.int64(out_offsets[r])
                    a_begin = types.int64(a_offsets[r])
                    a_len = types.int64(a_offsets[r + 1]) - a_begin
                    b_begin = types.int64(b_offsets[r])
                k = p - row_start
                if k < a_len:
                    out_chars[p] = a_chars[a_begin + k]
                else:
                    out_chars[p] = b_chars[b_begin + (k - a_len)]
                p += 1
        t0 += types.int64(cuda.gridDim.x) * tile


_kernel = cuda.jit(kernel)


def run(a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    _kernel[sms * BLOCKS_PER_SM, THREADS](a_offsets, a_chars, b_offsets, b_chars, out_offsets, out_chars, n)
