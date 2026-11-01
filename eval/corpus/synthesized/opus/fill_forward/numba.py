"""fill_forward: out[i] = values[j] for the largest valid row j <= i, or 0 when rows
0..i are all null.

A run of nulls can span the whole column, so the kernel is a cooperative launch (all
blocks resident, grid.sync between phases) and block b owns rows
[begin, end) = [b * chunk, (b + 1) * chunk):
  1. Each block finds its last valid row, scanning the validity bits backward from
     the end of its rows 256 rows at a time, and stores that row number (or -1) in
     out[end - 1] (overwritten in phase 3).
  2. Each block looks back over the previous blocks, 256 at a time; the nearest
     block with a valid row holds the last valid row before this block, and its
     value is the carry (0 when there is none).
  3. Each block scans its rows in tiles of 256, keeping the value of the last valid
     row (warp shuffles, then warp totals in shared memory), uses the carry for rows
     before the first valid row of the tile, and writes out.
"""

from numba import cuda, types

THREADS = 256
MAX_WARPS = 32
REDUCE_SLOTS = MAX_WARPS + 1  # one per warp, plus the result
FULL_MASK = 0xFFFFFFFF

TYPES = (
    types.int64[:],  # values
    types.uint8[:],  # valid
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


@cuda.jit(device=True, inline=True)
def _is_valid(valid, i):
    return ((valid[i >> 3] >> (i & 7)) & 1) != 0


def kernel(values, valid, out, n):
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

    # Phase 1: last valid row of this block's rows.
    if begin < end:
        last = types.int64(-1)
        t1 = end
        while t1 > begin and last < 0:
            i = t1 - 1 - tid
            cand = types.int64(-1)
            if i >= begin and _is_valid(valid, i):
                cand = i
            last = _block_max(cand, s_red)
            t1 -= nthreads
        if tid == 0:
            out[end - 1] = last
    grid.sync()

    # Phase 2: value of the last valid row before this block.
    carry = types.int64(0)
    if begin < end:
        # Every block before this one has a full chunk of rows, and later blocks
        # have later rows, so the largest stored row is the nearest one.
        k0 = b - 1
        while k0 >= 0:
            k = k0 - tid
            cand = types.int64(-1)
            if k >= 0:
                cand = out[(k + 1) * chunk - 1]
            last = _block_max(cand, s_red)
            if last >= 0:
                carry = values[last]
                break
            k0 -= nthreads
    grid.sync()

    # Phase 3: forward fill of this block's rows.
    t0 = begin
    while t0 < end:
        i = t0 + tid
        f = types.int32(0)
        v = types.int64(0)
        if i < end and _is_valid(valid, i):
            f = types.int32(1)
            v = values[i]
        d = 1
        while d < 32:
            of = cuda.shfl_up_sync(FULL_MASK, f, d)
            ov = cuda.shfl_up_sync(FULL_MASK, v, d)
            if lane >= d and f == 0:
                v = ov
                f = of
            d <<= 1
        if lane == 31:
            s_flag[warp] = f
            s_val[warp] = v
        cuda.syncthreads()
        # Take the last valid value of the warps before this one, then the carry.
        w = warp - 1
        while w >= 0 and f == 0:
            if s_flag[w] != 0:
                v = s_val[w]
                f = 1
            w -= 1
        if f == 0:
            v = carry
        if i < end:
            out[i] = v
        if tid == nthreads - 1:
            s_carry[0] = v
        cuda.syncthreads()
        carry = s_carry[0]
        t0 += nthreads


_kernel = cuda.jit(kernel)


def run(values, valid, out, n):
    args = (values, valid, out, n)
    compiled = _kernel.compile(tuple(_kernel.typeof_pyval(a) for a in args))
    resident = compiled.max_cooperative_grid_blocks(THREADS)
    blocks = max(1, min(resident, (n + THREADS - 1) // THREADS))
    _kernel[blocks, THREADS](*args)
