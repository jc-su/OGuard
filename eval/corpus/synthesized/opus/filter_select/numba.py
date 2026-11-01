"""filter_select: write the row ids i with quantity[i] < threshold into sel.

The kernel processes the rows in tiles of TILE rows. Each block reserves one
range of sel per tile with a single atomic add on count, so the kernel
requires count[0] == 0 on entry. run() sets count[0] to zero before the
launch, so a caller of run() can pass any initial value in count.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, float64, int32, int64, uint32
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256  # Threads per block. The kernel requires blockDim.x == BLOCK.
WARPS = BLOCK // 32
ITEMS = 4  # Rows per thread in one tile.
TILE = BLOCK * ITEMS
FULL_MASK = 0xFFFFFFFF

TYPES = (float64[:], float64, int32[:], int32[:], int64)


def kernel(
    quantity: npt.NDArray[np.float64],
    threshold: float,
    sel: npt.NDArray[np.int32],
    count: npt.NDArray[np.int32],
    n: int,
) -> None:
    warp_matches = cuda.shared.array(WARPS, int32)
    warp_slot = cuda.shared.array(WARPS, int32)
    masks = cuda.local.array(ITEMS, uint32)

    tid = cuda.threadIdx.x
    lane = cuda.laneid
    warp = tid // 32
    lanes_below = uint32((1 << lane) - 1)

    # The loop bounds are the same for all threads of a block, so every
    # thread reaches each syncthreads call.
    for tile in range(cuda.blockIdx.x * TILE, n, cuda.gridDim.x * TILE):
        # Evaluate the predicate and count the matches of each warp.
        matches = 0
        for k in range(ITEMS):
            i = tile + k * BLOCK + tid
            keep = i < n and quantity[i] < threshold
            masks[k] = cuda.ballot_sync(FULL_MASK, keep)
            matches += cuda.popc(masks[k])
        if lane == 0:
            warp_matches[warp] = matches
        cuda.syncthreads()

        # Reserve one range of sel for the tile and split it among the warps.
        if tid == 0:
            total = 0
            for w in range(WARPS):
                total += warp_matches[w]
            slot = cuda.atomic.add(count, 0, total)
            for w in range(WARPS):
                warp_slot[w] = slot
                slot += warp_matches[w]
        cuda.syncthreads()

        # Each matching lane writes its row id after the matches of the
        # lower lanes of its warp.
        slot = warp_slot[warp]
        for k in range(ITEMS):
            if (masks[k] >> lane) & 1:
                sel[slot + cuda.popc(masks[k] & lanes_below)] = tile + k * BLOCK + tid
            slot += cuda.popc(masks[k])
        # Do not overwrite the shared arrays before all warps read them.
        cuda.syncthreads()


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
    grid = min((n + TILE - 1) // TILE, resident)
    _launch[grid, BLOCK](quantity, threshold, sel, count, n)
