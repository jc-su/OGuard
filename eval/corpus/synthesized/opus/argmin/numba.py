"""argmin: row[0] = smallest i such that values[i] is the minimum of values; stays -1
for n = 0.

Each block reduces its rows to the best (value, row) pair, ordered by value and
then by row, and publishes it with a compare-and-swap loop on row[0]. The loop
compares against values[row[0]], so the published row only ever improves; a stale
read of row[0] only causes a failed CAS and a retry with the fresh value.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 8
FULL_MASK = 0xFFFFFFFF
INT64_MAX = 9223372036854775807
NO_ROW = INT64_MAX  # row sentinel: loses every tie

TYPES = (
    types.int64[:],  # values
    types.int32[:],  # row
    types.int64,  # n
)


@cuda.jit(device=True, inline=True)
def _better(v, r, bv, br):
    return v < bv or (v == bv and r < br)


def kernel(values, row, n):
    s_val = cuda.shared.array(32, types.int64)
    s_row = cuda.shared.array(32, types.int64)
    best_v = types.int64(INT64_MAX)
    best_r = types.int64(NO_ROW)
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        v = values[i]
        if _better(v, i, best_v, best_r):
            best_v = v
            best_r = i
        i += stride
    d = 16
    while d > 0:
        ov = cuda.shfl_xor_sync(FULL_MASK, best_v, d)
        orow = cuda.shfl_xor_sync(FULL_MASK, best_r, d)
        if _better(ov, orow, best_v, best_r):
            best_v = ov
            best_r = orow
        d >>= 1
    tid = cuda.threadIdx.x
    if tid & 31 == 0:
        s_val[tid >> 5] = best_v
        s_row[tid >> 5] = best_r
    cuda.syncthreads()
    if tid != 0:
        return
    for w in range(1, cuda.blockDim.x >> 5):
        if _better(s_val[w], s_row[w], best_v, best_r):
            best_v = s_val[w]
            best_r = s_row[w]
    if best_r == NO_ROW:
        return  # block had no rows

    cur = row[0]
    while True:
        if cur >= 0 and not _better(best_v, best_r, values[cur], cur):
            return
        old = cuda.atomic.cas(row, 0, cur, types.int32(best_r))
        if old == cur:
            return
        cur = old


_kernel = cuda.jit(kernel)


def run(values, row, n):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](values, row, n)
