"""Numba CUDA kernel: days_to_year.

Converts a day count since 1970-01-01 (proleptic Gregorian) into its
calendar year, for dates in years 1-9999. Uses Howard Hinnant's
`civil_from_days` algorithm (http://howardhinnant.github.io/date_algorithms.html),
which computes the year in O(1) integer arithmetic (no loops, no libc
date functions) from the days-since-epoch value; for years >= 1 its
"proleptic Gregorian, year 0 = 1 BC" numbering coincides with the usual
1-9999 numbering, so no offset is needed. One thread per row.
"""

from numba import cuda, int32, int64

TYPES = (int32[:], int32[:], int64)

THREADS_PER_BLOCK = 256


def kernel(days, year, n):
    i = cuda.grid(1)
    if i < n:
        z = int64(days[i]) + 719468
        era = (z if z >= 0 else z - 146096) // 146097
        doe = z - era * 146097  # [0, 146096]
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365  # [0, 399]
        y = yoe + era * 400
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)  # [0, 365]
        mp = (5 * doy + 2) // 153  # [0, 11]
        m = mp + (3 if mp < 10 else -9)  # [1, 12]
        year[i] = int32(y + (1 if m <= 2 else 0))


def run(days, year, n):
    if n <= 0:
        return
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    cuda.jit(kernel)[blocks, THREADS_PER_BLOCK](days, year, n)
