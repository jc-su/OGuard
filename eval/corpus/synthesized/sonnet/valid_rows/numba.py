"""Numba CUDA kernel: valid_rows.

Compacts the row ids whose bit is set in an Arrow validity bitmap into
`rows`, and writes how many there are into `count[0]`.

Approach: each thread tests one bit of the bitmap. Instead of one
atomicAdd per valid row (which would serialize on a single counter),
each warp reserves its output slots with a single atomicAdd covering all
of its valid lanes (warp-aggregated atomics), using ballot_sync + popc to
compute the intra-warp rank of each lane and shfl_sync to broadcast the
warp's base offset from the elected leader lane.
"""

from numba import cuda, int32, int64, uint8

TYPES = (uint8[:], int32[:], int32[:], int64)

THREADS_PER_BLOCK = 256


def kernel(valid, rows, count, n):
    i = cuda.grid(1)
    wants = False
    if i < n:
        byte = valid[i >> 3]
        wants = ((byte >> (i & 7)) & 1) != 0

    mask = cuda.activemask()
    vote = cuda.ballot_sync(mask, wants)
    if wants:
        lane = cuda.laneid
        lane_mask_lt = cuda.lanemask_lt()
        rank = cuda.popc(vote & lane_mask_lt)
        leader = cuda.ffs(vote) - 1
        warp_offset = int32(0)
        if lane == leader:
            warp_offset = cuda.atomic.add(count, 0, cuda.popc(vote))
        warp_offset = cuda.shfl_sync(mask, warp_offset, leader)
        rows[warp_offset + rank] = int32(i)


def run(valid, rows, count, n):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](valid, rows, count, n)
