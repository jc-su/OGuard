"""range_window_sum: out[i] = sum of values[j] over the rows j with
ts[i] - range <= ts[j] <= ts[i] (RANGE BETWEEN range PRECEDING AND CURRENT ROW),
modulo 2^64.

With P(q) = values[0] + ... + values[q - 1], out[i] = P(U_i) - P(L_i), where
L_i = first row with ts >= ts[i] - range and U_i = first row with ts > ts[i]. L_i and
U_i are found by galloping search from row i (cost logarithmic in the distance). A
window can cover the whole column, so the kernel is a cooperative launch (all
blocks resident, grid.sync between phases) and block b owns rows
[begin, end) = [b * chunk, (b + 1) * chunk):
  1. Each block sums its values and stores the sum in out[begin].
  2. Each block copies all block sums to shared memory and turns them into
     exclusive prefix sums, so P(k * chunk) is known for every block k. A grid.sync
     then lets phase 3 overwrite the stored sums.
  3. Each block evaluates P at the queries L_i and U_i of its rows, 256 rows per
     tile, with two cursors (position, P(position)). L_i and U_i are nondecreasing
     in i, so each cursor only moves forward: it adds values 256 at a time with a
     block-wide inclusive scan and each thread picks P at its query from the scan.
     The U cursor starts at P(begin); the L cursor starts at the block boundary
     below L_begin, so it reads at most one chunk before reaching it. In total a
     block reads O(chunk + its window span) values.
Sums use int64 arithmetic, which wraps modulo 2^64.
"""

from numba import cuda, types

THREADS = 256
MAX_BLOCKS = 1024
MAX_THREADS = 1024
MAX_WARPS = 32
FULL_MASK = 0xFFFFFFFF

TYPES = (
    types.int64[:],  # ts
    types.int64[:],  # values
    types.int64[:],  # out
    types.int64,  # n
    types.int64,  # range
)


@cuda.jit(device=True, inline=True)
def _block_inclusive_sum(v, s_warp):
    """Inclusive sum over the block and the block total."""
    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    d = 1
    while d < 32:
        o = cuda.shfl_up_sync(FULL_MASK, v, d)
        if lane >= d:
            v += o
        d <<= 1
    if lane == 31:
        s_warp[warp] = v
    cuda.syncthreads()
    before = types.int64(0)
    total = types.int64(0)
    for w in range(cuda.blockDim.x >> 5):
        if w < warp:
            before += s_warp[w]
        total += s_warp[w]
    cuda.syncthreads()
    return v + before, total


@cuda.jit(device=True, inline=True)
def _block_max(v, s):
    d = 16
    while d > 0:
        v = max(v, cuda.shfl_xor_sync(FULL_MASK, v, d))
        d >>= 1
    tid = cuda.threadIdx.x
    if tid & 31 == 0:
        s[tid >> 5] = v
    cuda.syncthreads()
    m = s[0]
    for w in range(1, cuda.blockDim.x >> 5):
        m = max(m, s[w])
    cuda.syncthreads()
    return m


@cuda.jit(device=True, inline=True)
def _lower_bound_back(ts, i, x):
    """First j in [0, i] with ts[j] >= x, given ts[i] >= x."""
    hi = i
    step = types.int64(1)
    lo = hi - step
    while lo >= 0 and ts[lo] >= x:
        hi = lo
        step <<= 1
        lo = hi - step
    lo = max(lo, types.int64(-1)) + 1
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if ts[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cuda.jit(device=True, inline=True)
def _upper_bound_fwd(ts, i, n, x):
    """First j in (i, n] with ts[j] > x (n when none), given ts[i] <= x."""
    lo = i
    step = types.int64(1)
    hi = lo + step
    while hi < n and ts[hi] <= x:
        lo = hi
        step <<= 1
        hi = lo + step
    hi = min(hi, n)
    lo += 1
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if ts[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cuda.jit(device=True, inline=True)
def _prefix_at(values, pos, val, q, q_max, s_warp, s_scan):
    """Moves the cursor (pos, val = P(pos)) to q_max (block-uniform) and returns
    (P(q), new pos, new val) for this thread's query q in [pos, q_max]."""
    tid = cuda.threadIdx.x
    result = val
    while pos < q_max:
        step = min(q_max - pos, types.int64(cuda.blockDim.x))
        x = types.int64(0)
        if tid < step:
            x = values[pos + tid]
        incl, total = _block_inclusive_sum(x, s_warp)
        s_scan[tid] = incl
        cuda.syncthreads()
        if q > pos and q <= pos + step:
            result = val + s_scan[q - pos - 1]
        cuda.syncthreads()
        pos += step
        val += total
    return result, pos, val


def kernel(ts, values, out, n, range):
    grid = cuda.cg.this_grid()
    s_prefix = cuda.shared.array(MAX_BLOCKS, types.int64)
    s_scan = cuda.shared.array(MAX_THREADS, types.int64)
    s_warp = cuda.shared.array(MAX_WARPS, types.int64)
    s_max = cuda.shared.array(MAX_WARPS, types.int64)
    s_low = cuda.shared.array(1, types.int64)

    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    blocks = types.int64(cuda.gridDim.x)
    b = types.int64(cuda.blockIdx.x)
    chunk = (n + blocks - 1) // blocks
    begin = min(b * chunk, n)
    end = min(begin + chunk, n)

    # Phase 1: sum of this block's values, stored in out[begin].
    s = types.int64(0)
    i = begin + tid
    while i < end:
        s += values[i]
        i += nthreads
    _, total = _block_inclusive_sum(s, s_warp)
    if tid == 0 and begin < end:
        out[begin] = total
    grid.sync()

    # Phase 2: s_prefix[k] = P(k * chunk) for every block k.
    per = (blocks + nthreads - 1) // nthreads
    # The argument `range` shadows the builtin, so the loops here are while loops.
    local = types.int64(0)
    k = tid * per
    while k < min((tid + 1) * per, blocks):
        v = types.int64(0)
        if k * chunk < n:
            v = out[k * chunk]
        s_prefix[k] = v
        local += v
        k += 1
    incl, total = _block_inclusive_sum(local, s_warp)
    running = incl - local
    k = tid * per
    while k < min((tid + 1) * per, blocks):
        v = s_prefix[k]
        s_prefix[k] = running
        running += v
        k += 1
    if tid == 0 and begin < end:
        s_low[0] = _lower_bound_back(ts, begin, ts[begin] - range)
    cuda.syncthreads()
    grid.sync()

    # Phase 3: out[i] = P(U_i) - P(L_i).
    if begin < end:
        u_pos = begin
        u_val = s_prefix[b]
        low_block = s_low[0] // chunk
        l_pos = low_block * chunk
        l_val = s_prefix[low_block]
        t0 = begin
        while t0 < end:
            i = t0 + tid
            active = i < end
            lq = types.int64(-1)
            uq = types.int64(-1)
            if active:
                x = ts[i]
                lq = _lower_bound_back(ts, i, x - range)
                uq = _upper_bound_fwd(ts, i, n, x)
            lq_max = _block_max(lq, s_max)
            uq_max = _block_max(uq, s_max)
            if not active:
                lq = l_pos
                uq = u_pos
            pl, l_pos, l_val = _prefix_at(values, l_pos, l_val, lq, lq_max, s_warp, s_scan)
            pu, u_pos, u_val = _prefix_at(values, u_pos, u_val, uq, uq_max, s_warp, s_scan)
            if active:
                out[i] = pu - pl
            t0 += nthreads


_kernel = cuda.jit(kernel)


def run(ts, values, out, n, range):
    args = (ts, values, out, n, range)
    compiled = _kernel.compile(tuple(_kernel.typeof_pyval(a) for a in args))
    resident = compiled.max_cooperative_grid_blocks(THREADS)
    blocks = max(1, min(resident, MAX_BLOCKS, (n + THREADS - 1) // THREADS))
    _kernel[blocks, THREADS](*args)
