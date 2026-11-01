"""radix_scatter: the scatter pass of an LSD radix sort on one byte (Numba CUDA).

Each block owns a tile of BLOCK * ITEMS consecutive rows and makes one
atomic add per nonzero digit on the global cursors, instead of one per row:

1. Rank: each row gets its rank among the tile's rows with the same digit.
   A warp groups its lanes by digit with match_any_sync, and the lowest lane
   of each group adds the group size to the block's shared-memory histogram,
   so a warp makes one shared atomic per distinct digit.
2. Reserve: thread d scans the histogram (the tile's local start of digit d)
   and reserves the digit's output range with one atomic add on cursors[d].
3. Scatter: the rows are first sorted by digit in shared memory, then written
   out in that order, so the stores of a warp go to a few contiguous runs.

The order within a digit is not stable (the task allows any order).

Bounds: every index is tested or clamped where it is used. A key index is
tested against n; a digit is in [0, 255] because it is masked with 255, and
the value RADIX that marks a row past the end is tested before it indexes; a
position in the sorted tile, which comes from shared memory, is clamped to
[0, TILE - 1]; an output position, which comes from the atomics on `cursors`,
is compared with n before the store. None of these tests changes a result for
input that meets the contract. The loops that index memory are while loops
with the bound in their condition.
"""

import numpy as np
from numba import cuda, int32, int64, uint8, uint32
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
WARPS = BLOCK // 32
ITEMS = 8
TILE = BLOCK * ITEMS
RADIX = 256  # one digit per thread in the reserve step: RADIX == BLOCK
FULL_MASK = 0xFFFFFFFF

TYPES = (uint32[:], int32[:], int32[:], int64, int64)


def kernel(keys: np.ndarray, cursors: np.ndarray, out: np.ndarray, n: int, shift: int) -> None:
    local_start = cuda.shared.array(RADIX, int32)
    out_offset = cuda.shared.array(RADIX, int32)
    warp_total = cuda.shared.array(WARPS, int32)
    stage_row = cuda.shared.array(TILE, int32)
    stage_digit = cuda.shared.array(TILE, uint8)
    digits = cuda.local.array(ITEMS, int32)
    ranks = cuda.local.array(ITEMS, int32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    tile_start = int64(cuda.blockIdx.x) * TILE
    tile_len = min(n - tile_start, TILE)  # rows of the tile (none past the end)

    # 1. Rank within the tile. local_start holds the histogram here.
    local_start[tid] = 0
    cuda.syncthreads()
    below = cuda.lanemask_lt()
    k = 0
    while k < ITEMS:
        i = tile_start + k * BLOCK + tid
        digit = int32(RADIX)  # rows past the end form their own group and are skipped
        if i < n:
            digit = int32((keys[i] >> shift) & 255)
        peers = cuda.match_any_sync(FULL_MASK, digit)
        leader = cuda.ffs(peers) - 1
        rank = int32(0)
        if lane == leader and digit < RADIX:
            rank = cuda.atomic.add(local_start, digit & 255, cuda.popc(peers))
        digits[k] = digit
        ranks[k] = cuda.shfl_sync(FULL_MASK, rank, leader) + cuda.popc(peers & below)
        k += 1
    cuda.syncthreads()

    # 2. Exclusive scan of the histogram and one global reservation per digit.
    c = local_start[tid]
    s = c
    offset = 1
    while offset < 32:
        t = cuda.shfl_up_sync(FULL_MASK, s, offset)
        if lane >= offset:
            s += t
        offset <<= 1
    if lane == 31:
        warp_total[warp] = s
    cuda.syncthreads()
    start = s - c
    w = 0
    while w < warp:
        start += warp_total[w]
        w += 1
    reserved = int32(0)
    if c > 0:
        reserved = cuda.atomic.add(cursors, tid, c)
    out_offset[tid] = reserved - start
    local_start[tid] = start
    cuda.syncthreads()

    # 3. Sort the tile by digit in shared memory, then write it out in order.
    k = 0
    while k < ITEMS:
        digit = digits[k]
        if digit < RADIX:
            pos = min(max(local_start[digit & 255] + ranks[k], 0), TILE - 1)
            stage_row[pos] = tile_start + k * BLOCK + tid
            stage_digit[pos] = digit
        k += 1
    cuda.syncthreads()
    x = tid
    while x < tile_len:
        o = int64(out_offset[stage_digit[x]]) + x
        if o >= 0 and o < n:
            out[o] = stage_row[x]
        x += BLOCK


_kernel = cuda.jit(kernel)


def run(
    keys: DeviceNDArray, cursors: DeviceNDArray, out: DeviceNDArray, n: int, shift: int
) -> None:
    if n == 0:
        return
    grid = (n + TILE - 1) // TILE
    _kernel[grid, BLOCK](keys, cursors, out, n, shift)
