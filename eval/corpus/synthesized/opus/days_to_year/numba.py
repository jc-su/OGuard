"""days_to_year: the proleptic Gregorian year of each day count since 1970-01-01 (Numba CUDA).

The civil-from-days algorithm (H. Hinnant) restricted to the year: shift the
epoch to 0000-03-01 so that the leap day ends the year, split the day count
into 400-year eras of 146097 days, and find the year of the era from the day
of the era. A date in January or February belongs to the next civil year.

For dates in the years 1 to 9999, the shifted day count z is in
[306, 3652424], so all the arithmetic is unsigned 32-bit and every division is
by a constant (a multiply-high). Each intermediate is cast back to uint32 so
that the compiler keeps 32-bit operations.

One thread per row. The kernel has no grid-stride loop because LLVM unrolls
such a loop behind a 64-bit division that costs more than the row itself; run
launches ceil(n / 256) blocks, which covers every row.
"""

import numpy as np
from numba import cuda, int32, int64, uint32
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCK = 256

TYPES = (int32[:], int32[:], int64)


def kernel(days: np.ndarray, year: np.ndarray, n: int) -> None:
    i = cuda.grid(1)
    if i < n:
        z = uint32(days[i] + 719468)
        era = uint32(z // uint32(146097))
        doe = uint32(z - era * uint32(146097))
        yoe = uint32(
            uint32(doe - doe // uint32(1460) + doe // uint32(36524) - doe // uint32(146096))
            // uint32(365)
        )
        doy = uint32(doe - uint32(uint32(365) * yoe + yoe // uint32(4) - yoe // uint32(100)))
        y = uint32(yoe + era * uint32(400))
        if doy >= uint32(306):
            y = uint32(y + uint32(1))
        year[i] = y


_kernel = cuda.jit(kernel)


def run(days: DeviceNDArray, year: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    grid = (n + BLOCK - 1) // BLOCK
    _kernel[grid, BLOCK](days, year, n)
