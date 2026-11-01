"""block_sort: sort the keys of each tile of 1024 rows ascending.

One block of 256 threads sorts one tile with a bitonic sort in shared memory.
A short last tile is padded with INT32_MAX, which sorts after every real key.
"""

import numba
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TILE = 1024
THREADS = 256
PAIRS = TILE // 2
INT32_MAX = 2147483647


def kernel(
    keys: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    # numba-cuda's stubs type blockIdx.x and threadIdx.x as properties; on the device they are ints.
    tile: int = cuda.blockIdx.x  # ty: ignore[invalid-assignment]
    if tile >= (n + TILE - 1) // TILE:
        return
    tid: int = cuda.threadIdx.x  # ty: ignore[invalid-assignment]
    base = tile * TILE
    buf = cuda.shared.array(TILE, numba.int32)
    for j in range(tid, TILE, THREADS):
        r = base + j
        if r < n:
            buf[j] = keys[r]
        else:
            buf[j] = INT32_MAX
    cuda.syncthreads()

    k = 2
    while k <= TILE:
        j = k >> 1
        while j >= 1:
            for p in range(tid, PAIRS, THREADS):
                i = (p // j) * (j << 1) + (p & (j - 1))
                partner = i | j
                x = buf[i]
                y = buf[partner]
                if ((i & k) == 0) == (x > y):
                    buf[i] = y
                    buf[partner] = x
            cuda.syncthreads()
            j >>= 1
        k <<= 1

    for j in range(tid, TILE, THREADS):
        r = base + j
        if r < n:
            out[r] = buf[j]


TYPES = (
    types.int32[:],
    types.int32[:],
    types.int64,
)

_jitted = cuda.jit(kernel)


def run(
    keys: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    blocks = max(1, (n + TILE - 1) // TILE)
    _jitted[blocks, THREADS](keys, out, n)
