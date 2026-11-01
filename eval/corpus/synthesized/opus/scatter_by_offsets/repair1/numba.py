"""Scatter by offsets: write the row id of each flagged row to its offset.

`offsets` is the exclusive prefix sum of `flags`, so with flags of 0 or 1 the
flagged rows land in `out[0:total]` in row order. This is the scatter step of
stream compaction.

`total` is the length of `out`. The kernel writes a row id only when its
offset is in [0, total). For offsets that are the exclusive prefix sum of
flags of 0 or 1, every flagged offset is in that range, so the check skips no
write. The check keeps each store inside `out` for any value in `offsets`.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, types
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.dispatcher import CUDADispatcher

BLOCK = 256
BLOCKS_PER_SM = 8


def kernel(
    flags: npt.NDArray[np.int8],
    offsets: npt.NDArray[np.int32],
    out: npt.NDArray[np.int32],
    n: int,
    total: int,
) -> None:
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    while i < n:
        if flags[i] != 0:
            position = offsets[i]
            if 0 <= position and position < total:
                out[position] = types.int32(i)
        i += stride


TYPES: tuple[types.Type, ...] = (
    types.int8[:],
    types.int32[:],
    types.int32[:],
    types.int64,
    types.int64,
)

_kernel: CUDADispatcher = cuda.jit(kernel)


def run(
    flags: DeviceNDArray,
    offsets: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
    total: int,
) -> None:
    """Scatter the flagged row ids of rows [0, n) into out[0:total]."""
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    grid = max(1, min((n + BLOCK - 1) // BLOCK, BLOCKS_PER_SM * sms))
    _kernel[grid, BLOCK](flags, offsets, out, n, total)
