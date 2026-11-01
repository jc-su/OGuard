"""radix_scatter: the scatter pass of an LSD radix sort on one byte (Numba CUDA).

Each block owns a tile of BLOCK * ITEMS consecutive rows and makes one
atomic add per nonzero digit on the global cursors, instead of one per row:

1. Rank: each row gets its rank among the tile's rows with the same digit.
   A warp groups its lanes by digit with match_any_sync, and the lowest lane
   of each group adds the group size to the block's shared-memory histogram,
   so a warp makes one shared atomic per distinct digit. The row's digit and
   rank go to shared memory, packed as digit << 16 | rank.
2. Reserve: thread d scans the histogram (the tile's local start of digit d)
   and reserves the digit's output range with one atomic add on cursors[d].
3. Scatter: the rows are first sorted by digit in shared memory, then written
   out in that order, so the stores of a warp go to a few contiguous runs.

The order within a digit is not stable (the task allows any order).

Bounds: every access is proved by the code at the access alone, without a
minimum, a maximum or a loop invariant. A key is read and a row id stored
only under the two comparisons 0 <= index and index < n of the very index
used; an output position comes from the atomics on `cursors`. An index into
the 256-entry arrays (cursors, the histogram and the output offsets) is
masked with 255, the warp index with WARPS - 1, and an index into a
tile-sized shared array with TILE - 1 (TILE is a power of two). A loop over
the tile takes its position through that mask too, so that the compiler forms
each address from the tested index instead of a pointer that steps with the
loop; a loop with a fixed count advances its slot with the mask
(slot = (slot + BLOCK) & mask) for the same reason. None of these changes a
result for input that meets the contract and a launch with BLOCK threads per
block. The per-row digits and ranks live in
shared memory, so the kernel keeps no per-thread array in local memory.
"""

import numpy as np
from numba import cuda, int32, int64, uint8, uint32
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
WARPS = BLOCK // 32
ITEMS = 8
TILE = BLOCK * ITEMS
LAST = TILE - 1  # the mask of a tile-sized shared-array index; TILE is a power of two
RADIX = 256  # one digit per thread in the reserve step: RADIX == BLOCK
FULL_MASK = 0xFFFFFFFF

TYPES = (uint32[:], int32[:], int32[:], int64, int64)


def kernel(keys: np.ndarray, cursors: np.ndarray, out: np.ndarray, n: int, shift: int) -> None:
    local_start = cuda.shared.array(RADIX, int32)
    out_offset = cuda.shared.array(RADIX, int32)
    warp_total = cuda.shared.array(WARPS, int32)
    digit_rank = cuda.shared.array(TILE, int32)
    stage_row = cuda.shared.array(TILE, int32)
    stage_digit = cuda.shared.array(TILE, uint8)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    bin = tid & (RADIX - 1)  # the digit of this thread in the reserve step
    tile_start = int64(cuda.blockIdx.x) * TILE
    tile_len = min(n - tile_start, TILE)  # rows of the tile (none past the end)

    # 1. Rank within the tile. local_start holds the histogram here.
    local_start[bin] = 0
    cuda.syncthreads()
    below = cuda.lanemask_lt()
    s = tid & LAST  # the row's place in the tile, k * BLOCK + tid
    k = 0
    while k < ITEMS:
        i = tile_start + s
        digit = int32(RADIX)  # rows past the end form their own group and are skipped
        if i >= 0 and i < n:
            digit = int32((keys[i] >> shift) & 255)
        peers = cuda.match_any_sync(FULL_MASK, digit)
        leader = cuda.ffs(peers) - 1
        rank = int32(0)
        if lane == leader and digit < RADIX:
            rank = cuda.atomic.add(local_start, digit & (RADIX - 1), cuda.popc(peers))
        rank = cuda.shfl_sync(FULL_MASK, rank, leader) + cuda.popc(peers & below)
        digit_rank[s] = (digit << 16) | rank
        s = (s + BLOCK) & LAST
        k += 1
    cuda.syncthreads()

    # 2. Exclusive scan of the histogram and one global reservation per digit.
    c = local_start[bin]
    s = c
    offset = 1
    while offset < 32:
        t = cuda.shfl_up_sync(FULL_MASK, s, offset)
        if lane >= offset:
            s += t
        offset <<= 1
    if lane == 31:
        warp_total[warp & (WARPS - 1)] = s
    cuda.syncthreads()
    start = s - c
    w = 0
    while w < warp:
        start += warp_total[w & (WARPS - 1)]
        w += 1
    reserved = int32(0)
    if c > 0:
        reserved = cuda.atomic.add(cursors, bin, c)
    out_offset[bin] = reserved - start
    local_start[bin] = start
    cuda.syncthreads()

    # 3. Sort the tile by digit in shared memory, then write it out in order.
    s = tid & LAST
    k = 0
    while k < ITEMS:
        v = digit_rank[s]
        digit = v >> 16
        if digit < RADIX:
            pos = (local_start[digit & (RADIX - 1)] + (v & 0xFFFF)) & LAST
            stage_row[pos] = tile_start + s
            stage_digit[pos] = digit
        s = (s + BLOCK) & LAST
        k += 1
    cuda.syncthreads()
    x = tid
    while x < tile_len:
        s = x & LAST
        o = int64(out_offset[stage_digit[s]]) + s
        if o >= 0 and o < n:
            out[o] = stage_row[s]
        x += BLOCK


_kernel = cuda.jit(kernel)


def run(
    keys: DeviceNDArray, cursors: DeviceNDArray, out: DeviceNDArray, n: int, shift: int
) -> None:
    if n == 0:
        return
    grid = (n + TILE - 1) // TILE
    _kernel[grid, BLOCK](keys, cursors, out, n, shift)
