import numpy as np
from numba import cuda
from numba.types import int32, int64


def kernel(days, months, out, n):
    i = np.int64(cuda.blockIdx.x) * 256 + np.int64(cuda.threadIdx.x)
    if i < n:
        # Civil date from days since 1970-01-01. z counts from 0000-03-01 and is
        # nonnegative for the years 1 to 9999, so floor and truncating division agree.
        z = np.int64(days[i]) + 719468
        era = z // 146097
        doe = z - era * 146097
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        y = yoe + era * 400
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        d = doy - (153 * mp + 2) // 5 + 1
        if mp < 10:
            m = mp + 3
        else:
            m = mp - 9
        if m <= 2:
            y += 1

        total = y * 12 + (m - 1) + np.int64(months[i])
        ty = total // 12
        tm = total % 12 + 1

        if tm == 2:
            if (ty % 4 == 0 and ty % 100 != 0) or ty % 400 == 0:
                dim = 29
            else:
                dim = 28
        elif tm == 4 or tm == 6 or tm == 9 or tm == 11:
            dim = 30
        else:
            dim = 31
        if d > dim:
            d = dim

        # Days since 1970-01-01 from the civil date (ty, tm, d).
        y2 = ty
        if tm <= 2:
            y2 -= 1
        era2 = y2 // 400
        yoe2 = y2 - era2 * 400
        if tm > 2:
            mp2 = tm - 3
        else:
            mp2 = tm + 9
        doy2 = (153 * mp2 + 2) // 5 + d - 1
        doe2 = yoe2 * 365 + yoe2 // 4 - yoe2 // 100 + doy2
        out[i] = np.int32(era2 * 146097 + doe2 - 719468)


TYPES = (int32[:], int32[:], int32[:], int64)

_kernel = cuda.jit(kernel)


def run(days, months, out, n):
    grid = max(1, (n + 255) // 256)
    _kernel[grid, 256](days, months, out, n)
