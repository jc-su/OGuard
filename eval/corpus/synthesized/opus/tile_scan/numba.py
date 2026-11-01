"""tile_scan: exclusive prefix sum within tiles of 1024 consecutive rows.

out[i] is the sum of values[j] over the rows j < i of row i's tile, and
totals[t] is the sum of tile t. One block of 256 threads scans one tile: the
block loads the tile into shared memory with coalesced reads, each thread sums
its 4 consecutive rows, a warp shuffle scan and a scan of the 8 warp sums give
each thread its prefix, and the block writes the tile back with coalesced
stores. Sums wrap around in 32-bit two's complement arithmetic.
"""

import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
TILE = 1024
ITEMS = TILE // THREADS
WARPS = THREADS // 32
# One pad word after every 32 words puts the rows 4t + j that the 32 threads
# of a warp read at once in distinct shared memory banks.
PADDED_TILE = TILE + TILE // 32
FULL_MASK = 0xFFFFFFFF


def kernel(
    values: DeviceNDArray,
    out: DeviceNDArray,
    totals: DeviceNDArray,
    n: int,
) -> None:
    tile = cuda.shared.array(PADDED_TILE, types.int32)
    warp_scan = cuda.shared.array(WARPS, types.int32)

    # Numba reads these special registers as integers inside a kernel. Its
    # Python stubs declare them as properties, which the type checker rejects.
    t: int = cuda.threadIdx.x  # ty: ignore[invalid-assignment]
    block: int = cuda.blockIdx.x  # ty: ignore[invalid-assignment]
    lane = t & 31
    warp = t >> 5
    base = np.int64(block) * TILE
    count = min(n - base, TILE)

    # Coalesced load of the tile. Rows past the end of the input count as 0.
    for j in range(ITEMS):
        e = t + j * THREADS
        if e < count:
            tile[e + (e >> 5)] = values[base + e]
        else:
            tile[e + (e >> 5)] = 0
    cuda.syncthreads()

    # Thread t owns the tile rows 4t .. 4t + 3, stored from index 4t + t / 8.
    row = ITEMS * t + (t >> 3)
    x0 = tile[row]
    x1 = tile[row + 1]
    x2 = tile[row + 2]
    x3 = tile[row + 3]
    thread_sum = np.int32(x0 + x1 + x2 + x3)

    # Inclusive scan of the thread sums within each warp.
    inclusive = thread_sum
    d = 1
    while d < 32:
        up = cuda.shfl_up_sync(FULL_MASK, inclusive, d)
        if lane >= d:
            inclusive = np.int32(inclusive + up)
        d <<= 1
    if lane == 31:
        warp_scan[warp] = inclusive
    cuda.syncthreads()

    # The first warp replaces the warp sums by their inclusive scan.
    if warp == 0:
        total = warp_scan[lane] if lane < WARPS else np.int32(0)
        d = 1
        while d < WARPS:
            up = cuda.shfl_up_sync(FULL_MASK, total, d)
            if lane >= d:
                total = np.int32(total + up)
            d <<= 1
        if lane < WARPS:
            warp_scan[lane] = total
    cuda.syncthreads()

    # Exclusive prefix of each of the thread's rows, written back in place.
    running = np.int32(inclusive - thread_sum)
    if warp > 0:
        running = np.int32(running + warp_scan[warp - 1])
    tile[row] = running
    running = np.int32(running + x0)
    tile[row + 1] = running
    running = np.int32(running + x1)
    tile[row + 2] = running
    running = np.int32(running + x2)
    tile[row + 3] = running
    if t == 0:
        totals[block] = warp_scan[WARPS - 1]
    cuda.syncthreads()

    # Coalesced store of the rows that exist.
    for j in range(ITEMS):
        e = t + j * THREADS
        if e < count:
            out[base + e] = tile[e + (e >> 5)]


TYPES = (
    types.int32[:],
    types.int32[:],
    types.int32[:],
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    values: DeviceNDArray,
    out: DeviceNDArray,
    totals: DeviceNDArray,
    n: int,
) -> None:
    tiles = (n + TILE - 1) // TILE
    if tiles > 0:
        _kernel[tiles, THREADS](values, out, totals, n)
