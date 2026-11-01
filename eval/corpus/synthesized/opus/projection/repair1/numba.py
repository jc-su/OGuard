"""projection: out[i] = price[i] * (1 - discount[i]) for i in [0, n).

The grid-stride loop is a while loop that compares the row index with n
before each access. A range() loop compiles to a trip-count loop, which does
not show that bound in the code.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, float64, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256  # Threads per block.

TYPES = (float64[:], float64[:], float64[:], int64)


def kernel(
    price: npt.NDArray[np.float64],
    discount: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
    n: int,
) -> None:
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        out[i] = price[i] * (1.0 - discount[i])
        i += stride


_launch = cuda.jit(TYPES)(kernel)


def run(
    price: DeviceNDArray,
    discount: DeviceNDArray,
    out: DeviceNDArray,
    n: int,
) -> None:
    """Write price[i] * (1 - discount[i]) into out[i] for every i in [0, n)."""
    if price.shape != (n,) or discount.shape != (n,) or out.shape != (n,):
        raise ValueError(
            f"expected price, discount, and out of shape ({n},), "
            f"got {price.shape}, {discount.shape}, {out.shape}"
        )
    if n == 0:
        return
    device = cuda.get_current_device()
    resident = device.MULTIPROCESSOR_COUNT * (device.MAX_THREADS_PER_MULTIPROCESSOR // BLOCK)
    grid = min((n + BLOCK - 1) // BLOCK, resident)
    _launch[grid, BLOCK](price, discount, out, n)
