"""Prefix offsets: offsets[0] = 0 and offsets[i + 1] = offsets[i] + lengths[i]."""

from numba import cuda, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int32[:], int32[:], int64)

THREADS = 256
WARPS = THREADS // 32
ITEMS = 16
TILE = THREADS * ITEMS
# One padding word per 32 words keeps the per-thread reads free of bank conflicts.
PADDED_TILE = TILE + TILE // 32
FULL_MASK = 0xFFFFFFFF


def kernel(lengths, offsets, n):
    # The signature has no scratch buffer to pass a carry between blocks, so
    # block 0 scans the whole column tile by tile and the other blocks have no
    # work. The kernel expects blockDim.x == THREADS.
    if cuda.blockIdx.x != 0:
        return
    buf = cuda.shared.array(PADDED_TILE, int32)
    warp_sums = cuda.shared.array(WARPS, int64)
    loaded = cuda.local.array(ITEMS, int32)
    scanned = cuda.local.array(ITEMS, int64)
    t = cuda.threadIdx.x
    lane = cuda.laneid
    warp = t // 32
    if t == 0:
        offsets[0] = 0

    # Coalesced load of the first tile.
    for q in range(ITEMS):
        row = t + q * THREADS
        loaded[q] = 0
        if row < n:
            loaded[q] = lengths[row]
    carry = 0
    base = 0
    while base < n:
        for q in range(ITEMS):
            j = t + q * THREADS
            buf[j + (j >> 5)] = loaded[q]
        cuda.syncthreads()

        # Issue the loads of the next tile; they complete while this tile is scanned.
        for q in range(ITEMS):
            row = base + TILE + t + q * THREADS
            loaded[q] = 0
            if row < n:
                loaded[q] = lengths[row]

        # Each thread scans its ITEMS consecutive elements in registers.
        total = 0
        for q in range(ITEMS):
            j = t * ITEMS + q
            total += buf[j + (j >> 5)]
            scanned[q] = total

        # Inclusive scan of the thread totals over the warp, then over the warps.
        inclusive = total
        d = 1
        while d < 32:
            up = cuda.shfl_up_sync(FULL_MASK, inclusive, d)
            if lane >= d:
                inclusive += up
            d *= 2
        if lane == 31:
            warp_sums[warp] = inclusive
        cuda.syncthreads()
        if warp == 0:
            warp_total = 0
            if lane < WARPS:
                warp_total = warp_sums[lane]
            d = 1
            while d < WARPS:
                up = cuda.shfl_up_sync(FULL_MASK, warp_total, d)
                if lane >= d:
                    warp_total += up
                d *= 2
            if lane < WARPS:
                warp_sums[lane] = warp_total
        cuda.syncthreads()

        prefix = carry + inclusive - total
        if warp > 0:
            prefix += warp_sums[warp - 1]
        for q in range(ITEMS):
            j = t * ITEMS + q
            buf[j + (j >> 5)] = scanned[q] + prefix
        carry += warp_sums[WARPS - 1]
        cuda.syncthreads()

        # Coalesced store of the tile.
        for q in range(ITEMS):
            j = t + q * THREADS
            row = base + j
            if row < n:
                offsets[row + 1] = buf[j + (j >> 5)]
        # buf and warp_sums are written again by the next tile.
        cuda.syncthreads()
        base += TILE


_KERNEL = cuda.jit(kernel)


def run(lengths: DeviceNDArray, offsets: DeviceNDArray, n: int) -> None:
    _KERNEL[1, THREADS](lengths, offsets, n)
