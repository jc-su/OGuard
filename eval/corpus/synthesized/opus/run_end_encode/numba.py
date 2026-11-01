"""run_end_encode: Arrow run-end encoding of values. count[0] = number of runs r; for
run j, run_ends[j] = row after its last row and run_values[j] = its value.

Row i ends a run when i == n - 1 or values[i] != values[i + 1]; run j is the run
ending at the (j + 1)-th such row. The kernel is a cooperative launch (all blocks
resident, grid.sync between phases) and block b owns rows [b * chunk, (b + 1) * chunk):
  1. Each block counts the run ends in its rows. A block with c > 0 run ends takes a
     slot from the counter count[0] (zero on entry) and records (b, c) there in
     run_ends / run_values. At most min(blocks, r) slots are used, all below r, so
     the positions at r and above are never touched.
  2. Each block reads the recorded slots and sums the counts of the blocks before it
     (its first output position) and of all blocks (r).
  3. Block 0 sets count[0] = r, and each block writes its runs at its first output
     position plus their rank among the run ends of its rows (a block-wide scan of
     ballots per tile of 256 rows).
"""

from numba import cuda, types

THREADS = 256
MAX_WARPS = 32
REDUCE_SLOTS = MAX_WARPS + 1  # one per warp, plus the result
FULL_MASK = 0xFFFFFFFF

TYPES = (
    types.int64[:],  # values
    types.int32[:],  # run_ends
    types.int64[:],  # run_values
    types.int32[:],  # count
    types.int64,  # n
)


@cuda.jit(device=True, inline=True)
def _is_run_end(values, i, n):
    return i == n - 1 or values[i] != values[i + 1]


@cuda.jit(device=True, inline=True)
def _block_sum(v, s):
    """Sum of v over the block, in every thread. s holds MAX_WARPS + 1 entries."""
    d = 16
    while d > 0:
        v += cuda.shfl_xor_sync(FULL_MASK, v, d)
        d >>= 1
    tid = cuda.threadIdx.x
    if tid & 31 == 0:
        s[tid >> 5] = v
    cuda.syncthreads()
    if tid == 0:
        t = types.int64(0)
        for w in range(cuda.blockDim.x >> 5):
            t += s[w]
        s[MAX_WARPS] = t
    cuda.syncthreads()
    total = s[MAX_WARPS]
    cuda.syncthreads()
    return total


def kernel(values, run_ends, run_values, count, n):
    grid = cuda.cg.this_grid()
    s_red = cuda.shared.array(REDUCE_SLOTS, types.int64)
    s_warp = cuda.shared.array(MAX_WARPS, types.int32)
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    b = types.int64(cuda.blockIdx.x)
    chunk = (n + cuda.gridDim.x - 1) // cuda.gridDim.x
    begin = min(b * chunk, n)
    end = min(begin + chunk, n)

    # Phase 1: run ends in [begin, end).
    c = types.int64(0)
    i = begin + tid
    while i < end:
        if _is_run_end(values, i, n):
            c += 1
        i += nthreads
    c = _block_sum(c, s_red)
    if tid == 0 and c > 0:
        slot = cuda.atomic.add(count, 0, 1)
        run_ends[slot] = b
        run_values[slot] = c
    grid.sync()

    # Phase 2: run ends before this block, and in total.
    used = count[0]
    before = types.int64(0)
    total = types.int64(0)
    j = tid
    while j < used:
        cj = run_values[j]
        total += cj
        if run_ends[j] < b:
            before += cj
        j += nthreads
    before = _block_sum(before, s_red)
    total = _block_sum(total, s_red)
    grid.sync()

    # Phase 3: write the runs.
    if b == 0 and tid == 0:
        count[0] = total
    lane_mask = cuda.lanemask_lt()
    warp = tid >> 5
    pos = before
    t0 = begin
    while t0 < end:
        i = t0 + tid
        flag = i < end and _is_run_end(values, i, n)
        ballot = cuda.ballot_sync(FULL_MASK, flag)
        if tid & 31 == 0:
            s_warp[warp] = cuda.popc(ballot)
        cuda.syncthreads()
        rank = cuda.popc(ballot & lane_mask)
        tile_total = 0
        for w in range(nthreads >> 5):
            if w < warp:
                rank += s_warp[w]
            tile_total += s_warp[w]
        if flag:
            run_ends[pos + rank] = i + 1
            run_values[pos + rank] = values[i]
        pos += tile_total
        cuda.syncthreads()
        t0 += nthreads


_kernel = cuda.jit(kernel)


def run(values, run_ends, run_values, count, n):
    args = (values, run_ends, run_values, count, n)
    compiled = _kernel.compile(tuple(_kernel.typeof_pyval(a) for a in args))
    resident = compiled.max_cooperative_grid_blocks(THREADS)
    blocks = max(1, min(resident, (n + THREADS - 1) // THREADS))
    _kernel[blocks, THREADS](*args)
