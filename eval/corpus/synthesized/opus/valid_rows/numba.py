"""valid_rows: the ids of the valid rows of an Arrow validity bitmap (Numba CUDA).

Each block owns a tile of TILE = 2048 consecutive rows; each warp owns 256 of
them, which are 32 bytes of the bitmap: lane l reads byte l once. In step j
(0 to 7) the warp takes bit j of every lane's byte, the rows 8 * l + j, and a
ballot turns them into a 32-bit mask, so a lane's output slot is the popcount
of the mask bits below it. The warp counts go through a block-level exclusive
scan, and one atomic add on `count` per block reserves the block's output
range. The stores of a step go to consecutive slots, so they coalesce. The
order of the ids is not sorted (the task allows any order).
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
    warp_offset = cuda.shared.array(WARPS, int32)
    block_base = cuda.shared.array(1, int32)
    masks = cuda.local.array(8, int32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = tid >> 5
    row0 = int64(cuda.blockIdx.x) * TILE + tid * 8  # the row of bit 0 of this lane's byte

    bits = int32(0)
    if row0 < n:
        bits = int32(valid[row0 >> 3])
        if row0 + 8 > n:
            bits &= (1 << (n - row0)) - 1  # the padding bits past row n
    total = 0
    for j in range(8):
        mask = cuda.ballot_sync(FULL_MASK, ((bits >> j) & 1) != 0)
        masks[j] = mask
        total += cuda.popc(mask)
    if lane == 0:
        warp_offset[warp] = total
    cuda.syncthreads()

    if tid == 0:
        running = 0
        for w in range(WARPS):
            c = warp_offset[w]
            warp_offset[w] = running
            running += c
        base = 0
        if running > 0:
            base = cuda.atomic.add(count, 0, running)
        block_base[0] = base
    cuda.syncthreads()

    pos = block_base[0] + warp_offset[warp]
    below = cuda.lanemask_lt()
    for j in range(8):
        mask = masks[j]
        if (bits >> j) & 1:
            rows[pos + cuda.popc(mask & below)] = row0 + j
        pos += cuda.popc(mask)


_kernel = cuda.jit(kernel)


def run(valid: DeviceNDArray, rows: DeviceNDArray, count: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    grid = (n + TILE - 1) // TILE
    _kernel[grid, BLOCK](valid, rows, count, n)
