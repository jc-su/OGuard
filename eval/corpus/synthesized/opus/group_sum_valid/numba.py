"""group_sum_valid: counts[k] += number of valid rows of group k, sums[k] += their values
(modulo 2^64).

g <= 1024, so each block accumulates into a shared-memory table of g counters and
sums and adds the nonzero entries to global memory once at the end. A warp whose
32 rows all fall in one group (sorted input or few groups) first reduces them with
shuffles and issues one shared atomic instead of 32.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 8
MAX_GROUPS = 1024
FULL_MASK = 0xFFFFFFFF

TYPES = (
    types.int32[:],  # group
    types.int64[:],  # value
    types.uint8[:],  # valid
    types.int64[:],  # counts
    types.int64[:],  # sums
    types.int64,  # n
    types.int64,  # g
)


def kernel(group, value, valid, counts, sums, n, g):
    s_count = cuda.shared.array(MAX_GROUPS, types.int64)
    s_sum = cuda.shared.array(MAX_GROUPS, types.int64)
    tid = cuda.threadIdx.x
    k = tid
    while k < g:
        s_count[k] = 0
        s_sum[k] = 0
        k += cuda.blockDim.x
    cuda.syncthreads()

    lane = tid & 31
    stride = cuda.gridsize(1)
    # base is the same for all lanes of a warp, so the loop is warp-uniform.
    base = cuda.grid(1) - lane
    while base < n:
        i = base + lane
        in_range = i < n
        own = types.int32(0)
        if in_range:
            own = group[i]
        # Lanes past the end take lane 0's group (row base < n) and contribute nothing.
        lane0 = cuda.shfl_sync(FULL_MASK, own, 0)
        grp = own if in_range else lane0
        use = in_range and ((valid[i >> 3] >> (i & 7)) & 1) != 0
        c = types.int64(0)
        v = types.int64(0)
        if use:
            c = types.int64(1)
            v = value[i]
        if cuda.all_sync(FULL_MASK, grp == cuda.shfl_sync(FULL_MASK, grp, 0)):
            d = 16
            while d > 0:
                c += cuda.shfl_xor_sync(FULL_MASK, c, d)
                v += cuda.shfl_xor_sync(FULL_MASK, v, d)
                d >>= 1
            if lane == 0 and c != 0:
                cuda.atomic.add(s_count, grp, c)
                cuda.atomic.add(s_sum, grp, v)
        elif use:
            cuda.atomic.add(s_count, grp, 1)
            cuda.atomic.add(s_sum, grp, v)
        base += stride
    cuda.syncthreads()

    k = tid
    while k < g:
        if s_count[k] != 0:
            cuda.atomic.add(counts, k, s_count[k])
            cuda.atomic.add(sums, k, s_sum[k])
        k += cuda.blockDim.x


_kernel = cuda.jit(kernel)


def run(group, value, valid, counts, sums, n, g):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](group, value, valid, counts, sums, n, g)
