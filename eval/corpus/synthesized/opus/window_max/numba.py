"""Window max: out[i] is the largest of values[max(0, i - w + 1) : i + 1]."""

import numpy as np
from numba import cuda, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int64[:], int64[:], int64, int64)

THREADS = 256
MAX_W = 1024
# Outputs per block tile. A tile reads OUT_TILE + w - 1 values.
OUT_TILE = 1024
SPAN = OUT_TILE + MAX_W
INT64_MIN = np.iinfo(np.int64).min


def kernel(values, out, n, w):
    # Each block computes OUT_TILE outputs at a time from the OUT_TILE + w - 1
    # values they cover, held in shared memory. Pass s turns maxima over 2^s
    # values into maxima over 2^(s + 1) values, until the length K is the
    # largest power of two not above w. A window of w values is then the union
    # of the K values at its start and the K values at its end.
    buf = cuda.shared.array((2, SPAN), int64)
    t = cuda.threadIdx.x
    step = cuda.blockDim.x
    span = OUT_TILE + w - 1
    ntiles = (n + OUT_TILE - 1) // OUT_TILE
    tile = cuda.blockIdx.x
    while tile < ntiles:
        base = tile * OUT_TILE
        # Local position 0 is row base - w + 1. Rows outside [0, n) hold INT64_MIN,
        # which never changes a maximum.
        j = t
        while j < span:
            row = base - w + 1 + j
            value = INT64_MIN
            if row >= 0 and row < n:
                value = values[row]
            buf[0, j] = value
            j += step
        cuda.syncthreads()

        src = 0
        k = 1
        while 2 * k <= w:
            # buf[src, j] is the maximum of local [j, j + k) for j <= span - k.
            j = t
            while j <= span - 2 * k:
                a = buf[src, j]
                b = buf[src, j + k]
                buf[1 - src, j] = a if a > b else b
                j += step
            cuda.syncthreads()
            src = 1 - src
            k *= 2

        # The window of row base + o is local [o, o + w - 1].
        o = t
        while o < OUT_TILE:
            row = base + o
            if row < n:
                a = buf[src, o]
                b = buf[src, o + w - k]
                out[row] = a if a > b else b
            o += step
        # buf is written again by the next tile.
        cuda.syncthreads()
        tile += cuda.gridDim.x


_KERNEL = cuda.jit(kernel)


def run(values: DeviceNDArray, out: DeviceNDArray, n: int, w: int) -> None:
    if not 1 <= w <= MAX_W:
        raise ValueError(f"w must be in [1, {MAX_W}], got {w}")
    if n == 0:
        return
    blocks = (n + OUT_TILE - 1) // OUT_TILE
    _KERNEL[blocks, THREADS](values, out, n, w)
