"""filter_select: sel[0:count] = row ids i with quantity[i] < threshold.

Warp-aggregated compaction. Each thread walks the rows with a grid stride.
On a selected row, the thread reads the active mask, the lowest active lane
does a single atomicAdd on count for all active lanes, and every active lane
writes its row id at base + (number of active lanes below it). Under
independent thread scheduling the active mask can cover a subset of the
lanes that hold a selected row; each subset then reserves its own slots,
which is still correct.

The store sits inside the row loop, whose condition i < n bounds the stored
row id on every path to the store. The output position is formed from base
as an unsigned value and compared against n before the store, so every store
stays inside sel (n slots) whatever count holds on entry. Under the contract
(count zero on entry) the comparison always holds, because at most n rows
are selected.

count must be zero on entry.
"""

from numba import cuda, float64, int32, int64, uint32

THREADS = 256
MAX_BLOCKS = 2048


def kernel(quantity, threshold, sel, count, n):
    lanemask_lt = cuda.lanemask_lt()
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    while i < n:
        if quantity[i] < threshold:
            active = cuda.activemask()
            rank = cuda.popc(active & lanemask_lt)
            base = int32(0)
            if rank == 0:
                base = cuda.atomic.add(count, 0, cuda.popc(active))
            base = cuda.shfl_sync(active, base, cuda.ffs(active) - 1)
            pos = int64(uint32(base)) + rank
            if pos < n:
                sel[pos] = i
        i += stride


TYPES = (float64[:], float64, int32[:], int32[:], int64)

_kernel = cuda.jit(kernel)


def run(quantity, threshold, sel, count, n):
    blocks = max(1, min((n + THREADS - 1) // THREADS, MAX_BLOCKS))
    _kernel[blocks, THREADS](quantity, threshold, sel, count, n)
