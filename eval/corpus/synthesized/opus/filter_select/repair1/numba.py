"""filter_select: write the row ids i with quantity[i] < threshold into sel.

Each block processes one tile of blockDim.x rows per iteration of a
grid-stride loop, one row per thread. Lane 0 of each warp reserves the range
of its warp's matches within the tile with an atomic add on a shared counter,
thread 0 reserves the range of the tile in sel with a single atomic add on
count, and each matching thread writes its row id into that range. The kernel
requires count[0] == 0 on entry and a block size that is a multiple of 32.
run() sets count[0] to zero before the launch, so a caller of run() can pass
any initial value in count.

A thread writes its row id only when its position is in [0, n). If count[0]
is zero on entry, every position is below the number of matches, which is at
most n, so the check skips no write. The check keeps each store inside sel
for any value in count on entry.

The grid-stride loop is a while loop that compares the tile start with n
before each iteration. A range() loop compiles to a trip-count loop, which
does not show that bound in the code.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, float64, int32, int64, uint32
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256  # Threads per block. The kernel requires a multiple of 32.
FULL_MASK = 0xFFFFFFFF

TYPES = (float64[:], float64, int32[:], int32[:], int64)


def kernel(
    quantity: npt.NDArray[np.float64],
    threshold: float,
    sel: npt.NDArray[np.int32],
    count: npt.NDArray[np.int32],
    n: int,
) -> None:
    # The ty: ignore comments below: to a host type checker, the Numba CUDA
    # stubs (cuda.threadIdx.x, cuda.blockDim.x, cuda.atomic.add, cuda.popc)
    # are Python properties and Stub classes. Numba compiles them to device
    # integers and intrinsics.

    # tile_count[0] counts the matches of the current tile. tile_first[0] is
    # the position in sel of the first match of the current tile.
    tile_count = cuda.shared.array(1, int32)
    tile_first = cuda.shared.array(1, int32)

    tid = cuda.threadIdx.x
    lane = tid & 31  # ty: ignore[unsupported-operator]
    lanes_below = uint32((1 << lane) - 1)
    stride = cuda.gridDim.x * cuda.blockDim.x  # ty: ignore[unsupported-operator]

    if tid == 0:
        tile_count[0] = 0
    cuda.syncthreads()

    # tile is the same for all threads of a block, so every thread reaches
    # each syncthreads call. Thread 0 writes tile_count and tile_first only
    # between the two syncthreads calls of a tile, when no other thread reads
    # or writes them.
    tile = cuda.blockIdx.x * cuda.blockDim.x  # ty: ignore[unsupported-operator]
    while tile < n:
        i = tile + tid
        keep = i < n and quantity[i] < threshold
        mask = uint32(cuda.ballot_sync(FULL_MASK, keep) & FULL_MASK)

        # Lane 0 reserves the range of its warp within the tile.
        warp_first = int32(0)
        if lane == 0 and mask != 0:
            warp_first = cuda.atomic.add(tile_count, 0, int32(cuda.popc(mask)))  # ty: ignore[too-many-positional-arguments]
        warp_first = cuda.shfl_sync(FULL_MASK, warp_first, 0)
        cuda.syncthreads()

        # Thread 0 reserves the range of the tile in sel.
        if tid == 0:
            matches = tile_count[0]
            if matches != 0:
                tile_first[0] = cuda.atomic.add(count, 0, matches)  # ty: ignore[too-many-positional-arguments]
            tile_count[0] = 0
        cuda.syncthreads()

        # Each matching lane writes its row id after the matches of the
        # lower lanes of its warp.
        if keep:
            position = tile_first[0] + warp_first + cuda.popc(mask & lanes_below)  # ty: ignore[too-many-positional-arguments]
            if 0 <= position and position < n:
                sel[position] = int32(i)
        tile += stride


_launch = cuda.jit(TYPES)(kernel)


def run(
    quantity: DeviceNDArray,
    threshold: float,
    sel: DeviceNDArray,
    count: DeviceNDArray,
    n: int,
) -> None:
    """Write the selected row ids into sel and their number into count."""
    if quantity.shape != (n,) or sel.shape != (n,) or count.shape != (1,):
        raise ValueError(
            f"expected quantity and sel of shape ({n},) and count of shape (1,), "
            f"got {quantity.shape}, {sel.shape}, {count.shape}"
        )
    count.copy_to_device(np.zeros(1, dtype=np.int32))
    if n == 0:
        return
    device = cuda.get_current_device()
    resident = device.MULTIPROCESSOR_COUNT * (device.MAX_THREADS_PER_MULTIPROCESSOR // BLOCK)
    grid = min((n + BLOCK - 1) // BLOCK, resident)
    _launch[grid, BLOCK](quantity, threshold, sel, count, n)
