"""valid_rows: compact the ids of the rows whose Arrow validity bit is set.

One thread per row. Each warp ballots its valid lanes, warp 0 prefix-sums
the warp counts with a shuffle scan, lane 31 claims one range of `rows` with
a single atomic add on `count`, and every valid lane stores its id at
block base + warp offset + rank within the warp, so the stores of a warp are
contiguous. Output order is by block claim order, not by row id.

Revision: the byte index into `valid` is tested against the bitmap size
(n + 7) >> 3 before the load, the store position is tested against n, and
the cross-warp prefix sum is a shuffle scan instead of a loop over shared
memory. The tests always pass under the task's contract (row i < n has its
bit inside the bitmap; `count` is zero on entry, so every position is below
n); they state the bound of each access for the memory checker.
"""

from numba import cuda, int32, int64, uint8

TYPES = (uint8[:], int32[:], int32[:], int64)
THREADS = 256
WARPS = THREADS // 32


def kernel(valid, rows, count, n):
    warp_count = cuda.shared.array(32, int32)  # one slot per lane of warp 0; WARPS are used
    warp_offset = cuda.shared.array(32, int32)
    block_base = cuda.shared.array(1, int32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5
    i = cuda.blockIdx.x * THREADS + tid
    nbytes = (n + 7) >> 3
    ok = False
    if i < n:
        byte = i >> 3
        if byte < nbytes:
            ok = ((valid[byte] >> (i & 7)) & 1) == 1
    ballot = cuda.ballot_sync(0xFFFFFFFF, ok)
    if lane == 0:
        warp_count[wid] = cuda.popc(ballot)
    cuda.syncthreads()
    if tid < 32:
        c = 0
        if tid < WARPS:
            c = warp_count[tid]
        x = c
        off = 1
        while off < 32:
            y = cuda.shfl_up_sync(0xFFFFFFFF, x, off)
            if lane >= off:
                x += y
            off <<= 1
        warp_offset[tid] = x - c
        if tid == 31:
            base = 0
            if x > 0:
                base = cuda.atomic.add(count, 0, x)
            block_base[0] = base
    cuda.syncthreads()
    if ok:
        rank = cuda.popc(ballot & cuda.lanemask_lt())
        pos = block_base[0] + warp_offset[wid] + rank
        if pos >= 0 and pos < n:
            rows[pos] = i


_kernel = cuda.jit(kernel)


def run(valid, rows, count, n):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](valid, rows, count, n)
