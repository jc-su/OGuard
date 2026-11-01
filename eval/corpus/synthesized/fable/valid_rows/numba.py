"""valid_rows: compact the ids of the rows whose Arrow validity bit is set.

One thread per row. Each warp ballots its valid lanes, the block prefix-sums
the eight warp counts, claims one range of `rows` with a single atomic add on
`count`, and every valid lane stores its id at
block base + warp offset + rank within the warp, so the stores of a warp are
contiguous. Output order is by block claim order, not by row id.
"""

from numba import cuda, int32, int64, uint8

TYPES = (uint8[:], int32[:], int32[:], int64)
THREADS = 256


def kernel(valid, rows, count, n):
    warp_offset = cuda.shared.array(8, int32)
    block_base = cuda.shared.array(1, int32)
    tid = cuda.threadIdx.x
    lane = tid & 31
    wid = tid >> 5
    i = cuda.blockIdx.x * THREADS + tid
    ok = False
    if i < n:
        ok = ((valid[i >> 3] >> (i & 7)) & 1) == 1
    ballot = cuda.ballot_sync(0xFFFFFFFF, ok)
    if lane == 0:
        warp_offset[wid] = cuda.popc(ballot)
    cuda.syncthreads()
    if tid == 0:
        total = 0
        for w in range(8):
            c = warp_offset[w]
            warp_offset[w] = total
            total += c
        if total > 0:
            block_base[0] = cuda.atomic.add(count, 0, total)
        else:
            block_base[0] = 0
    cuda.syncthreads()
    if ok:
        rank = cuda.popc(ballot & cuda.lanemask_lt())
        rows[block_base[0] + warp_offset[wid] + rank] = i


_kernel = cuda.jit(kernel)


def run(valid, rows, count, n):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](valid, rows, count, n)
