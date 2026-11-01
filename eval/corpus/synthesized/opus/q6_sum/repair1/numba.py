"""q6_sum: the aggregate of TPC-H Q6.

result[0] += sum of price[i] * discount[i] over the rows i with
date_lo <= shipdate[i] < date_hi, disc_lo <= discount[i] <= disc_hi, and
quantity[i] < qty_max. Each warp adds its partial sum to a shared block sum
with one atomic add, and each block adds its block sum to result[0] with one
atomic add, so result[0] must be zero on entry. The kernel requires a block
size that is a multiple of 32.

The grid-stride loop is a while loop that compares the row index with n
before each access. A range() loop compiles to a trip-count loop, which does
not show that bound in the code.
"""

import numpy as np
import numpy.typing as npt
from numba import cuda, float64, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256  # Threads per block. The kernel requires a multiple of 32.
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
    # The ty: ignore comments below: to a host type checker, the Numba CUDA
    # stubs (cuda.threadIdx.x, cuda.atomic.add) are Python properties and
    # Stub classes. Numba compiles them to device integers and atomic
    # instructions.
    block_sum = cuda.shared.array(1, float64)
    tid = cuda.threadIdx.x
    if tid == 0:
        block_sum[0] = 0.0

    acc = 0.0
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
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
        i += stride

    # Sum the values of the threads of each warp into lane 0.
    offset = 16
    while offset > 0:
        acc += cuda.shfl_down_sync(FULL_MASK, acc, offset)
        offset //= 2

    # Thread 0 has set block_sum to zero before any warp adds to it.
    cuda.syncthreads()
    if (tid & 31) == 0:  # ty: ignore[unsupported-operator]
        cuda.atomic.add(block_sum, 0, acc)  # ty: ignore[too-many-positional-arguments]
    cuda.syncthreads()
    if tid == 0:
        cuda.atomic.add(result, 0, block_sum[0])  # ty: ignore[too-many-positional-arguments]


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
