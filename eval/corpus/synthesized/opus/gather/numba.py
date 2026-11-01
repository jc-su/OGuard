"""gather: out[i] = col[sel[i]] for i in [0, m).

Every value of sel must be a row id of col, that is, in [0, n). The kernel
does not read n; the argument gives the length of col.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, float64, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256  # Threads per block.

TYPES = (float64[:], int32[:], float64[:], int64, int64)


def kernel(
    col: npt.NDArray[np.float64],
    sel: npt.NDArray[np.int32],
    out: npt.NDArray[np.float64],
    m: int,
    n: int,
) -> None:
    for i in range(cuda.grid(1), m, cuda.gridsize(1)):
        out[i] = col[sel[i]]


_launch = cuda.jit(TYPES)(kernel)


def run(
    col: DeviceNDArray,
    sel: DeviceNDArray,
    out: DeviceNDArray,
    m: int,
    n: int,
) -> None:
    """Write col[sel[i]] into out[i] for every i in [0, m)."""
    if col.shape != (n,) or sel.shape != (m,) or out.shape != (m,):
        raise ValueError(
            f"expected col of shape ({n},) and sel and out of shape ({m},), "
            f"got {col.shape}, {sel.shape}, {out.shape}"
        )
    if m == 0:
        return
    device = cuda.get_current_device()
    resident = device.MULTIPROCESSOR_COUNT * (device.MAX_THREADS_PER_MULTIPROCESSOR // BLOCK)
    grid = min((m + BLOCK - 1) // BLOCK, resident)
    _launch[grid, BLOCK](col, sel, out, m, n)
