"""tile_scan: exclusive prefix sum within tiles of 1024 rows, plus each tile's total.

One block of 256 threads handles one tile; each thread holds 4 consecutive rows.
"""

import numba
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TILE = 1024
THREADS = 256
PER_THREAD = TILE // THREADS
WARPS = THREADS // 32
FULL_MASK = 0xFFFFFFFF


def kernel(
    values: DeviceNDArray,
    out: DeviceNDArray,
    totals: DeviceNDArray,
    n: int,
) -> None:
    # numba-cuda's stubs type blockIdx.x and threadIdx.x as properties; on the device they are ints.
    tile: int = cuda.blockIdx.x  # ty: ignore[invalid-assignment]
    if tile >= (n + TILE - 1) // TILE:
        return
    tid: int = cuda.threadIdx.x  # ty: ignore[invalid-assignment]
    lane = tid & 31
    warp = tid >> 5
    base = tile * TILE + tid * PER_THREAD

    v0 = numba.int32(0)
    v1 = numba.int32(0)
    v2 = numba.int32(0)
    v3 = numba.int32(0)
    if base < n:
        v0 = values[base]
    if base + 1 < n:
        v1 = values[base + 1]
    if base + 2 < n:
        v2 = values[base + 2]
    if base + 3 < n:
        v3 = values[base + 3]
    local = v0 + v1 + v2 + v3

    # Inclusive scan of the per-thread sums within each warp.
    x = local
    for d in (1, 2, 4, 8, 16):
        y = cuda.shfl_up_sync(numba.uint32(FULL_MASK), x, d)
        if lane >= d:
            x += y

    # Inclusive scan of the warp totals by warp 0.
    warp_sums = cuda.shared.array(WARPS, numba.int32)
    if lane == 31:
        warp_sums[warp] = x
    cuda.syncthreads()
    if warp == 0:
        w = numba.int32(0)
        if lane < WARPS:
            w = warp_sums[lane]
        for d in (1, 2, 4):
            y = cuda.shfl_up_sync(numba.uint32(FULL_MASK), w, d)
            if lane >= d:
                w += y
        if lane < WARPS:
            warp_sums[lane] = w
    cuda.syncthreads()

    prefix = x - local
    if warp > 0:
        prefix += warp_sums[warp - 1]
    if base < n:
        out[base] = prefix
    if base + 1 < n:
        out[base + 1] = prefix + v0
    if base + 2 < n:
        out[base + 2] = prefix + v0 + v1
    if base + 3 < n:
        out[base + 3] = prefix + v0 + v1 + v2
    if tid == THREADS - 1:
        totals[tile] = prefix + local


TYPES = (
    types.int32[:],
    types.int32[:],
    types.int32[:],
    types.int64,
)

_jitted = cuda.jit(kernel)


def run(
    values: DeviceNDArray,
    out: DeviceNDArray,
    totals: DeviceNDArray,
    n: int,
) -> None:
    blocks = max(1, (n + TILE - 1) // TILE)
    _jitted[blocks, THREADS](values, out, totals, n)
