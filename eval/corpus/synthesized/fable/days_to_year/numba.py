"""days_to_year: year of a date given as days since 1970-01-01.

One thread per row, branch-free integer arithmetic (Howard Hinnant's
civil_from_days): shift the epoch to 0000-03-01, split into 400-year eras
and the day of era, derive the year of era and the day of the March-based
year, and add one when that day falls in January or February.
"""

from numba import cuda, int32, int64

TYPES = (int32[:], int32[:], int64)
THREADS = 256


def kernel(days, year, n):
    i = cuda.grid(1)
    if i >= n:
        return
    z = days[i] + 719468  # days since 0000-03-01
    era = z // 146097  # floor division, z may be negative
    doe = z - era * 146097  # day of era, 0 to 146096
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365  # 0 to 399
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)  # day of March-based year, 0 to 365
    y = yoe + era * 400
    if doy >= 306:  # January or February of the next civil year
        y += 1
    year[i] = y


_kernel = cuda.jit(kernel)


def run(days, year, n):
    blocks = max(1, (n + THREADS - 1) // THREADS)
    _kernel[blocks, THREADS](days, year, n)
