"""Transpose: out[j * rows + i] = source[i * cols + j] for a row-major matrix."""

from numba import cuda, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int32[:], int32[:], int64, int64)

THREADS = 256
TILE = 32
# The extra column keeps the column reads of the tile free of bank conflicts.
TILE_STRIDE = TILE + 1


def kernel(source, out, rows, cols):
    # Each block moves 32 x 32 tiles through shared memory so that both the
    # reads of source and the writes of out are coalesced.
    tile = cuda.shared.array((TILE, TILE_STRIDE), int32)
    tx = cuda.threadIdx.x % TILE
    ty = cuda.threadIdx.x // TILE
    step = cuda.blockDim.x // TILE
    tiles_c = (cols + TILE - 1) // TILE
    ntiles = ((rows + TILE - 1) // TILE) * tiles_c
    b = cuda.blockIdx.x
    while b < ntiles:
        r0 = (b // tiles_c) * TILE
        c0 = (b % tiles_c) * TILE
        c = c0 + tx
        y = ty
        while y < TILE:
            r = r0 + y
            if r < rows and c < cols:
                tile[y, tx] = source[r * cols + c]
            y += step
        cuda.syncthreads()
        r = r0 + tx
        y = ty
        while y < TILE:
            c = c0 + y
            if c < cols and r < rows:
                out[c * rows + r] = tile[tx, y]
            y += step
        # The tile is written again in the next iteration.
        cuda.syncthreads()
        b += cuda.gridDim.x


_KERNEL = cuda.jit(kernel)


def run(source: DeviceNDArray, out: DeviceNDArray, rows: int, cols: int) -> None:
    ntiles = ((rows + TILE - 1) // TILE) * ((cols + TILE - 1) // TILE)
    if ntiles == 0:
        return
    _KERNEL[ntiles, THREADS](source, out, rows, cols)
