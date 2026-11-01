"""Month start: the day number of the first day of the month that contains each day."""

from numba import cuda, int32, int64
from numba.cuda.cudadrv.devicearray import DeviceNDArray

TYPES = (int32[:], int32[:], int64)

THREADS = 256


def kernel(days, out, n):
    stride = cuda.gridsize(1)
    i = cuda.grid(1)
    while i < n:
        # Civil date from day number (H. Hinnant), with years starting on March 1
        # so that the leap day is the last day of the year. // is floor division.
        day = days[i]
        z = day + 719468
        era = z // 146097
        doe = z - era * 146097
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        # doy - (153 * mp + 2) // 5 is the zero-based day of the month.
        out[i] = day - doy + (153 * mp + 2) // 5
        i += stride


_KERNEL = cuda.jit(kernel)


def run(days: DeviceNDArray, out: DeviceNDArray, n: int) -> None:
    if n == 0:
        return
    blocks = (n + THREADS - 1) // THREADS
    _KERNEL[blocks, THREADS](days, out, n)
