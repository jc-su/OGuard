"""q6_sum: the aggregate of TPC-H Q6.

result[0] += sum of price[i] * discount[i] over the rows i with
date_lo <= shipdate[i] < date_hi, disc_lo <= discount[i] <= disc_hi, and
quantity[i] < qty_max. Each block adds its partial sum to result[0] with one
atomic add, so result[0] must be zero on entry. The kernel requires
blockDim.x == BLOCK.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, float64, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256  # Threads per block. The kernel requires blockDim.x == BLOCK.
WARPS = BLOCK // 32
FULL_MASK = 0xFFFFFFFF

TYPES = (
    int32[:],
    float64[:],
    float64[:],
    float64[:],
    int32,
    int32,
    float64,
    float64,
    float64,
    float64[:],
    int64,
)


def kernel(
    shipdate: npt.NDArray[np.int32],
    discount: npt.NDArray[np.float64],
    quantity: npt.NDArray[np.float64],
    price: npt.NDArray[np.float64],
    date_lo: int,
    date_hi: int,
    disc_lo: float,
    disc_hi: float,
    qty_max: float,
    result: npt.NDArray[np.float64],
    n: int,
) -> None:
    warp_sums = cuda.shared.array(WARPS, float64)

    acc = 0.0
    for i in range(cuda.grid(1), n, cuda.gridsize(1)):
        day = shipdate[i]
        disc = discount[i]
        if (
            date_lo <= day
            and day < date_hi
            and disc_lo <= disc
            and disc <= disc_hi
            and quantity[i] < qty_max
        ):
            acc += price[i] * disc

    # Sum the values of the threads of each warp into lane 0.
    offset = 16
    while offset > 0:
        acc += cuda.shfl_down_sync(FULL_MASK, acc, offset)
        offset //= 2

    lane = cuda.laneid
    warp = cuda.threadIdx.x // 32
    if lane == 0:
        warp_sums[warp] = acc
    cuda.syncthreads()

    # Warp 0 sums the warp sums and adds the block sum to the result.
    if warp == 0:
        acc = warp_sums[lane] if lane < WARPS else 0.0
        offset = 16
        while offset > 0:
            acc += cuda.shfl_down_sync(FULL_MASK, acc, offset)
            offset //= 2
        if lane == 0:
            cuda.atomic.add(result, 0, acc)


_launch = cuda.jit(TYPES)(kernel)


def run(
    shipdate: DeviceNDArray,
    discount: DeviceNDArray,
    quantity: DeviceNDArray,
    price: DeviceNDArray,
    date_lo: int,
    date_hi: int,
    disc_lo: float,
    disc_hi: float,
    qty_max: float,
    result: DeviceNDArray,
    n: int,
) -> None:
    """Add the Q6 revenue of the n rows to result[0], which is zero on entry."""
    columns = (shipdate, discount, quantity, price)
    if any(c.shape != (n,) for c in columns) or result.shape != (1,):
        raise ValueError(
            f"expected columns of shape ({n},) and result of shape (1,), "
            f"got {[c.shape for c in columns]}, {result.shape}"
        )
    if n == 0:
        return
    device = cuda.get_current_device()
    resident = device.MULTIPROCESSOR_COUNT * (device.MAX_THREADS_PER_MULTIPROCESSOR // BLOCK)
    grid = min((n + BLOCK - 1) // BLOCK, resident)
    _launch[grid, BLOCK](
        shipdate,
        discount,
        quantity,
        price,
        date_lo,
        date_hi,
        disc_lo,
        disc_hi,
        qty_max,
        result,
        n,
    )
