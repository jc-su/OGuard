"""block_sort: sort the keys of each tile of 1024 consecutive rows ascending.

One block of 256 threads sorts one tile with a bitonic sorting network and
writes it to the same rows of out.

Thread t holds the tile rows 4t .. 4t + 3 in registers. Stage (size, stride)
of the network compares rows e and e + stride for each e with bit `stride`
clear, and orders the pair ascending when bit `size` of e is clear. A stage
runs where the two rows of each pair are:
  stride 1 and 2:    in the registers of one thread;
  stride 4 to 64:    in threads of one warp, exchanged by warp shuffles;
  stride 128 to 512: in different warps, compared in shared memory.

The rows past the end of a short last tile are filled with INT32_MAX, which
sorts them after the real keys; their values equal any INT32_MAX key, so the
first count rows of the sorted tile are the sorted keys.
"""

import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS = 256
TILE = 1024
ITEMS = TILE // THREADS  # Rows per thread.
PAIRS = TILE // 2 // THREADS  # Shared memory pairs per thread per stage.
BLOCK_STRIDE = ITEMS * 32  # Smallest stride that crosses warps.
# Row e is stored at e + e / 32. The pad word after every 32 words puts the
# rows 4t .. 4t + 3 that a warp reads or writes at once in distinct banks.
PADDED_TILE = TILE + TILE // 32
FULL_MASK = 0xFFFFFFFF
INT32_MAX = np.int32(2147483647)


def kernel(
    keys: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    tile = cuda.shared.array(PADDED_TILE, types.int32)

    # Numba reads these special registers as integers inside a kernel. Its
    # Python stubs declare them as properties, which the type checker rejects.
    t: int = cuda.threadIdx.x  # ty: ignore[invalid-assignment]
    block: int = cuda.blockIdx.x  # ty: ignore[invalid-assignment]
    base = np.int64(block) * TILE
    count = min(n - base, TILE)
    own = ITEMS * t + (t >> 3)  # Padded index of row 4t.

    # Coalesced load of the tile, then each thread takes its four rows.
    for j in range(ITEMS):
        e = t + j * THREADS
        if e < count:
            tile[e + (e >> 5)] = keys[base + e]
        else:
            tile[e + (e >> 5)] = INT32_MAX
    cuda.syncthreads()
    v0 = tile[own]
    v1 = tile[own + 1]
    v2 = tile[own + 2]
    v3 = tile[own + 3]

    size = 2
    while size <= TILE:
        stride = size >> 1

        if stride >= BLOCK_STRIDE:
            # Every thread has read its rows back from the last use of tile.
            cuda.syncthreads()
            tile[own] = v0
            tile[own + 1] = v1
            tile[own + 2] = v2
            tile[own + 3] = v3
            cuda.syncthreads()
            while stride >= BLOCK_STRIDE:
                for q in range(PAIRS):
                    p = t + q * THREADS
                    lo = 2 * p - (p & (stride - 1))
                    hi = lo + stride
                    a = tile[lo + (lo >> 5)]
                    b = tile[hi + (hi >> 5)]
                    if (lo & size) == 0:
                        tile[lo + (lo >> 5)] = min(a, b)
                        tile[hi + (hi >> 5)] = max(a, b)
                    else:
                        tile[lo + (lo >> 5)] = max(a, b)
                        tile[hi + (hi >> 5)] = min(a, b)
                cuda.syncthreads()
                stride >>= 1
            v0 = tile[own]
            v1 = tile[own + 1]
            v2 = tile[own + 2]
            v3 = tile[own + 3]

        # For stride >= 2 all four rows of the thread share bit `size`.
        ascending = ((ITEMS * t) & size) == 0
        while stride >= ITEMS:
            # The partner of row 4t + r is row 4(t ^ lane_mask) + r.
            lane_mask = stride // ITEMS
            keep_min = ((t & lane_mask) == 0) == ascending
            p0 = cuda.shfl_xor_sync(FULL_MASK, v0, lane_mask)
            p1 = cuda.shfl_xor_sync(FULL_MASK, v1, lane_mask)
            p2 = cuda.shfl_xor_sync(FULL_MASK, v2, lane_mask)
            p3 = cuda.shfl_xor_sync(FULL_MASK, v3, lane_mask)
            v0 = min(v0, p0) if keep_min else max(v0, p0)
            v1 = min(v1, p1) if keep_min else max(v1, p1)
            v2 = min(v2, p2) if keep_min else max(v2, p2)
            v3 = min(v3, p3) if keep_min else max(v3, p3)
            stride >>= 1
        if stride == 2:
            if ascending:
                v0, v2 = min(v0, v2), max(v0, v2)
                v1, v3 = min(v1, v3), max(v1, v3)
            else:
                v0, v2 = max(v0, v2), min(v0, v2)
                v1, v3 = max(v1, v3), min(v1, v3)
        if ascending:
            v0, v1 = min(v0, v1), max(v0, v1)
        else:
            v0, v1 = max(v0, v1), min(v0, v1)
        if ((ITEMS * t + 2) & size) == 0:
            v2, v3 = min(v2, v3), max(v2, v3)
        else:
            v2, v3 = max(v2, v3), min(v2, v3)
        size <<= 1

    # Each thread puts its rows back, then the block stores the tile coalesced.
    cuda.syncthreads()
    tile[own] = v0
    tile[own + 1] = v1
    tile[own + 2] = v2
    tile[own + 3] = v3
    cuda.syncthreads()
    for j in range(ITEMS):
        e = t + j * THREADS
        if e < count:
            out[base + e] = tile[e + (e >> 5)]


TYPES = (
    types.int32[:],
    types.int32[:],
    types.int64,
)

_kernel = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    tiles = (n + TILE - 1) // TILE
    if tiles > 0:
        _kernel[tiles, THREADS](keys, out, n)
