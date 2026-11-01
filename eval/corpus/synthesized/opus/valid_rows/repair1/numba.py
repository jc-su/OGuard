"""valid_rows: the ids of the valid rows of an Arrow validity bitmap (Numba CUDA).

Each block owns a tile of TILE = 2048 consecutive rows; each warp owns 256 of
them, which are 32 bytes of the bitmap: lane l reads byte l once. In step j
(0 to 7) the warp takes bit j of every lane's byte, the rows 8 * l + j, and a
ballot turns them into a 32-bit mask, so a lane's output slot is the popcount
of the mask bits below it. The warp counts go through a block-level exclusive
scan, and one atomic add on `count` per block reserves the block's output
range; warp 0 does the scan and the atomic with warp shuffles. The stores of a
step go to consecutive slots, so they coalesce. The order of the ids is not
sorted (the task allows any order).

Bounds: a thread reads its bitmap byte only when the byte is below the bitmap
length (n + 7) >> 3 and its first row is below n (the two tests are
equivalent; each states the bound in one form). A store slot comes from the
atomic on `count`, so it is compared with n before the store; with `count`
zero on entry every slot is below n, and that test never skips a row. The
ballots are taken again in the store pass instead of being kept in a
per-thread array, and the shared array of warp offsets has one entry per lane,
so any lane index is inside it.
"""

import numpy as np
from numba import cuda, int32, int64, uint8
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
WARPS = BLOCK // 32
TILE = BLOCK * 8  # one bitmap byte per thread
FULL_MASK = 0xFFFFFFFF

TYPES = (uint8[:], int32[:], int32[:], int64)


def kernel(valid: np.ndarray, rows: np.ndarray, count: np.ndarray, n: int) -> None:
    # Warp counts, then output offsets; one entry per lane of warp 0.
    warp_offset = cuda.shared.array(32, int32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    nbytes = (n + 7) >> 3  # the length of the bitmap
    byte = int64(cuda.blockIdx.x) * BLOCK + tid  # this lane's bitmap byte
    row0 = byte * 8  # the row of its bit 0

    bits = int32(0)
    if byte < nbytes and row0 < n:
        bits = int32(valid[byte])
        if row0 + 8 > n:
            bits &= (1 << (n - row0)) - 1  # the padding bits past row n
    total = 0
    for j in range(8):
        total += cuda.popc(cuda.ballot_sync(FULL_MASK, ((bits >> j) & 1) != 0))
    if lane == 0:
        warp_offset[warp] = total
    cuda.syncthreads()

    # Warp 0 scans the warp counts (lane w holds warp w) and reserves the
    # block's output range with one atomic add.
    if warp == 0:
        c = 0
        if lane < WARPS:
            c = warp_offset[lane]
        inclusive = c
        offset = 1
        while offset < WARPS:
            t = cuda.shfl_up_sync(FULL_MASK, inclusive, offset)
            if lane >= offset:
                inclusive += t
            offset <<= 1
        running = cuda.shfl_sync(FULL_MASK, inclusive, WARPS - 1)
        base = 0
        if lane == 0 and running > 0:
            base = cuda.atomic.add(count, 0, running)
        base = cuda.shfl_sync(FULL_MASK, base, 0)
        if lane < WARPS:
            warp_offset[lane] = base + inclusive - c
    cuda.syncthreads()

    pos = int64(warp_offset[warp])
    below = cuda.lanemask_lt()
    for j in range(8):
        mask = cuda.ballot_sync(FULL_MASK, ((bits >> j) & 1) != 0)
        if (bits >> j) & 1:
            slot = pos + cuda.popc(mask & below)
            if slot >= 0 and slot < n:
                rows[slot] = row0 + j
        pos += cuda.popc(mask)


_kernel = cuda.jit(kernel)


def run(valid: DeviceNDArray, rows: DeviceNDArray, count: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    grid = (n + TILE - 1) // TILE
    _kernel[grid, BLOCK](valid, rows, count, n)
