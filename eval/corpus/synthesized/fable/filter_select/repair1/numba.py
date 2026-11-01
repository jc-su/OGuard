"""filter_select: sel[0:count] = row ids i with quantity[i] < threshold.

Warp-aggregated compaction. Each warp ballots the predicate, lane 0 does a
single atomicAdd on count for the whole warp, and every selected lane writes
its row id at base + (number of selected lanes below it). The loop bound is
rounded up to a multiple of the warp size so that every ballot is a full warp.

The output position is formed from base as an unsigned value and compared
against n before the store, so every store stays inside sel (n slots)
whatever count holds on entry. Under the contract (count zero on entry) the
comparison always holds, because at most n rows are selected.

count must be zero on entry.
"""

from numba import cuda, float64, int32, int64, uint32

FULL_MASK = 0xFFFFFFFF
THREADS = 256
MAX_BLOCKS = 2048


def kernel(quantity, threshold, sel, count, n):
    lane = cuda.laneid
    stride = cuda.gridsize(1)
    n_round = (n + 31) & ~31
    i = cuda.grid(1)
    while i < n_round:
        pred = False
        if i < n:
            pred = quantity[i] < threshold
        ballot = cuda.ballot_sync(FULL_MASK, pred)
        if ballot != 0:
            base = int32(0)
            if lane == 0:
                base = cuda.atomic.add(count, 0, cuda.popc(ballot))
            base = cuda.shfl_sync(FULL_MASK, base, 0)
            if pred:
                pos = int64(uint32(base)) + cuda.popc(ballot & cuda.lanemask_lt())
                if pos < n:
                    sel[pos] = i
        i += stride


TYPES = (float64[:], float64, int32[:], int32[:], int64)

_kernel = cuda.jit(kernel)


def run(quantity, threshold, sel, count, n):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](quantity, threshold, sel, count, n)
