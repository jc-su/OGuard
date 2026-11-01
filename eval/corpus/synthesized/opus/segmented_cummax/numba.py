"""segmented_cummax: out[i] = max(values[first[i] .. i]), a running maximum that restarts
at every partition start (first[i] == i).

A partition can span the whole column, so the kernel is a cooperative launch (all
blocks resident, grid.sync between phases) and block b owns rows
[begin, end) = [b * chunk, (b + 1) * chunk):
  1. Each block computes the maximum of the partition part that ends its rows,
     values[max(first[end - 1], begin) .. end), and stores it in out[end - 1]
     (overwritten in phase 3).
  2. If its first row continues a partition (first[begin] < begin), a block looks
     back over the previous blocks, 256 at a time, and takes the maximum of their
     stored values up to the nearest block in which that partition starts
     (first[end_k - 1] >= begin_k). That is the carry into its rows.
  3. Each block scans its rows in tiles of 256 with a segmented max scan (warp
     shuffles, then warp totals in shared memory), combining rows before the first
     partition start of the tile with the carry, and writes out.
"""

from numba import cuda, types

THREADS = 256
MAX_WARPS = 32
REDUCE_SLOTS = MAX_WARPS + 1  # one per warp, plus the result
FULL_MASK = 0xFFFFFFFF
NEG_INF = -9223372036854775808

TYPES = (
    types.int64[:],  # values
    types.int32[:],  # first
    types.int64[:],  # out
    types.int64,  # n
)


@cuda.jit(device=True, inline=True)
def _block_max(v, s):
    """Maximum of v over the block, in every thread. s holds MAX_WARPS + 1 entries."""
    d = 16
    while d > 0:
        v = max(v, cuda.shfl_xor_sync(FULL_MASK, v, d))
        d >>= 1
    tid = cuda.threadIdx.x
    if tid & 31 == 0:
        s[tid >> 5] = v
    cuda.syncthreads()
    if tid == 0:
        t = s[0]
        for w in range(1, cuda.blockDim.x >> 5):
            t = max(t, s[w])
        s[MAX_WARPS] = t
    cuda.syncthreads()
    result = s[MAX_WARPS]
    cuda.syncthreads()
    return result


def kernel(values, first, out, n):
    grid = cuda.cg.this_grid()
    s_red = cuda.shared.array(REDUCE_SLOTS, types.int64)
    s_val = cuda.shared.array(MAX_WARPS, types.int64)
    s_flag = cuda.shared.array(MAX_WARPS, types.int32)
    s_carry = cuda.shared.array(1, types.int64)

    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    lane = tid & 31
    warp = tid >> 5
    b = types.int64(cuda.blockIdx.x)
    chunk = (n + cuda.gridDim.x - 1) // cuda.gridDim.x
    begin = min(b * chunk, n)
    end = min(begin + chunk, n)

    # Phase 1: maximum of the partition part that ends this block's rows.
    if begin < end:
        m = types.int64(NEG_INF)
        i = max(types.int64(first[end - 1]), begin) + tid
        while i < end:
            m = max(m, values[i])
            i += nthreads
        m = _block_max(m, s_red)
        if tid == 0:
            out[end - 1] = m
    grid.sync()

    # Phase 2: carry from the partition that continues into this block.
    carry = types.int64(NEG_INF)
    if begin < end and first[begin] < begin:
        # Every block before this one has a full chunk of rows.
        k0 = b - 1
        while k0 >= 0:
            k = k0 - tid
            stop = types.int64(-1)
            v = types.int64(NEG_INF)
            if k >= 0:
                v = out[(k + 1) * chunk - 1]
                if first[(k + 1) * chunk - 1] >= k * chunk:
                    stop = k
            stop = _block_max(stop, s_red)
            if k < stop:
                v = NEG_INF
            v = _block_max(v, s_red)
            carry = max(carry, v)
            if stop >= 0:
                break
            k0 -= nthreads
    grid.sync()

    # Phase 3: segmented max scan of this block's rows.
    t0 = begin
    while t0 < end:
        i = t0 + tid
        f = types.int32(1)
        v = types.int64(NEG_INF)
        if i < end:
            f = types.int32(1 if first[i] == i else 0)
            v = values[i]
        d = 1
        while d < 32:
            of = cuda.shfl_up_sync(FULL_MASK, f, d)
            ov = cuda.shfl_up_sync(FULL_MASK, v, d)
            if lane >= d:
                if f == 0:
                    v = max(v, ov)
                f |= of
            d <<= 1
        if lane == 31:
            s_flag[warp] = f
            s_val[warp] = v
        cuda.syncthreads()
        # Combine with the warps before this one, then with the carry.
        w = warp - 1
        while w >= 0 and f == 0:
            v = max(v, s_val[w])
            f = s_flag[w]
            w -= 1
        if f == 0:
            v = max(v, carry)
        if i < end:
            out[i] = v
        if tid == nthreads - 1:
            s_carry[0] = v
        cuda.syncthreads()
        carry = s_carry[0]
        t0 += nthreads


_kernel = cuda.jit(kernel)


def run(values, first, out, n):
    args = (values, first, out, n)
    compiled = _kernel.compile(tuple(_kernel.typeof_pyval(a) for a in args))
    resident = compiled.max_cooperative_grid_blocks(THREADS)
    blocks = max(1, min(resident, (n + THREADS - 1) // THREADS))
    _kernel[blocks, THREADS](*args)
