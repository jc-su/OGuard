"""group_argmax: best[k] = smallest row of group k whose value is the largest of the
group; stays -1 for a group without rows.

best[k] holds a row; candidates are ordered by (value desc, row asc) and the row is
replaced by compare-and-swap, comparing against value[best[k]], so best[k] only
improves. A thread first reads best[k] and skips the CAS when its candidate does
not win; a stale read only shows an older (worse) row, so the skip is safe. To cut
contention when many rows share a group (few groups, or sorted groups), each warp
groups its lanes by group id with match_any_sync. When the warp holds at most
MAX_WARP_GROUPS distinct groups, each group is reduced with shuffles and its leader
issues one update; otherwise every lane updates on its own.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32
MAX_WARP_GROUPS = 4
FULL_MASK = 0xFFFFFFFF
INT64_MIN = -9223372036854775808
INT64_MAX = 9223372036854775807

TYPES = (
    types.int32[:],  # group
    types.int64[:],  # value
    types.int32[:],  # best
    types.int64,  # n
    types.int64,  # g
)


@cuda.jit(device=True, inline=True)
def _wins(v, r, bv, br):
    return v > bv or (v == bv and r < br)


@cuda.jit(device=True, inline=True)
def _update(best, value, k, v, r):
    cur = best[k]
    while cur < 0 or _wins(v, r, value[cur], cur):
        old = cuda.atomic.cas(best, k, cur, r)
        if old == cur:
            return
        cur = old


def kernel(group, value, best, n, g):
    lane = cuda.threadIdx.x & 31
    stride = cuda.gridsize(1)
    # base is the same for all lanes of a warp, so the loop is warp-uniform.
    base = cuda.grid(1) - lane
    while base < n:
        i = base + lane
        in_range = i < n
        k = types.int32(-1)
        v = types.int64(0)
        if in_range:
            k = group[i]
            v = value[i]
        peers = cuda.match_any_sync(FULL_MASK, k)
        leader = lane == cuda.ffs(peers) - 1
        leaders = cuda.ballot_sync(FULL_MASK, leader)
        if cuda.popc(leaders) <= MAX_WARP_GROUPS:
            while leaders != 0:
                src = cuda.ffs(leaders) - 1
                leaders &= leaders - types.uint32(1)
                gk = cuda.shfl_sync(FULL_MASK, k, src)
                bv = types.int64(INT64_MIN)
                br = types.int64(INT64_MAX)
                if k == gk:
                    bv = v
                    br = i
                d = 16
                while d > 0:
                    ov = cuda.shfl_xor_sync(FULL_MASK, bv, d)
                    orow = cuda.shfl_xor_sync(FULL_MASK, br, d)
                    if _wins(ov, orow, bv, br):
                        bv = ov
                        br = orow
                    d >>= 1
                if lane == src and gk >= 0:
                    _update(best, value, gk, bv, types.int32(br))
        elif in_range:
            _update(best, value, k, v, types.int32(i))
        base += stride


_kernel = cuda.jit(kernel)


def run(group, value, best, n, g):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](group, value, best, n, g)
