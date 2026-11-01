"""valid_rows: the ids of the valid rows of an Arrow validity bitmap (Numba CUDA).

Each block owns a tile of TILE = 2048 consecutive rows; each warp owns 256 of
them. In step j (0 to 7) lane l of a warp takes the row 32 * j + l of the
warp's range, reads the bitmap byte row >> 3 (the lanes of a step read 4
consecutive bytes) and its bit row & 7, and a ballot turns the 32 bits into a
mask, so a lane's output slot is the popcount of the mask bits below it. The
warp counts go through a block-level exclusive scan, and one atomic add on
`count` per block reserves the block's output range; warp 0 does the scan and
the atomic with warp shuffles. The stores of a step go to consecutive slots,
so they coalesce, and the ids come out ascending within a block. The order
of the blocks is the order of their atomics (the task allows any order).

Bounds: every access is proved by the code at the access alone, without a
minimum, a maximum or a loop invariant. A thread reads the bitmap byte
row >> 3 only under 0 <= row and row < n for the row itself (the bitmap has
(n + 7) / 8 bytes, so row >> 3 is one of them), and under the same test of
the byte index against (n + 7) >> 3. A store slot comes from the atomic on
`count` and is stored only under 0 <= slot and slot < n; with `count` zero on
entry every slot is below n, and that test never skips a row. The per-step
bits of a thread are kept in the bits of one integer, and the shared array of
warp counts has one entry per lane, indexed by a lane or warp number masked
with 31.
"""

import numpy as np
from numba import cuda, int32, int64, uint8
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256
WARPS = BLOCK // 32
STEPS = 8  # rows per thread
TILE = BLOCK * STEPS
FULL_MASK = 0xFFFFFFFF

TYPES = (uint8[:], int32[:], int32[:], int64)


def kernel(valid: np.ndarray, rows: np.ndarray, count: np.ndarray, n: int) -> None:
    # Warp counts, then output offsets; one entry per lane of warp 0.
    warp_offset = cuda.shared.array(32, int32)

    tid = cuda.threadIdx.x
    lane = tid & 31
    warp = (tid >> 5) & 31
    nbytes = (n + 7) >> 3  # the length of the bitmap
    first_row = int64(cuda.blockIdx.x) * TILE + warp * (32 * STEPS) + lane

    bits = int32(0)  # bit j: the row first_row + 32 * j is valid
    total = 0
    j = 0
    while j < STEPS:
        row = first_row + 32 * j
        byte = row >> 3
        valid_bit = 0
        if row >= 0 and row < n and byte >= 0 and byte < nbytes:
            valid_bit = (int32(valid[byte]) >> (row & 7)) & 1
        bits |= valid_bit << j
        total += cuda.popc(cuda.ballot_sync(FULL_MASK, valid_bit != 0))
        j += 1
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
    j = 0
    while j < STEPS:
        valid_bit = (bits >> j) & 1
        mask = cuda.ballot_sync(FULL_MASK, valid_bit != 0)
        if valid_bit != 0:
            slot = pos + cuda.popc(mask & below)
            if slot >= 0 and slot < n:
                rows[slot] = first_row + 32 * j
        pos += cuda.popc(mask)
        j += 1


_kernel = cuda.jit(kernel)


def run(valid: DeviceNDArray, rows: DeviceNDArray, count: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    grid = (n + TILE - 1) // TILE
    _kernel[grid, BLOCK](valid, rows, count, n)
