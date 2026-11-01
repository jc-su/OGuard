"""add_months: out[i] = days[i] plus months[i] months, keeping the day of the month or
clamping it to the last day of the target month.

Dates convert with Howard Hinnant's civil_from_days / days_from_civil algorithms on
400-year eras. Every date is in the years 1 to 9999, so all the shifted day and
year counts are nonnegative and floor division equals truncation. The month
arithmetic counts months from year 0: target = y * 12 + (m - 1) + months.
"""

from numba import cuda, types

THREADS = 256
BLOCKS_PER_SM = 32

TYPES = (
    types.int32[:],  # days
    types.int32[:],  # months
    types.int32[:],  # out
    types.int64,  # n
)


@cuda.jit(device=True, inline=True)
def _days_in_month(y, m):
    if m == 2:
        if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0:
            return 29
        return 28
    return 30 + ((m + (m >> 3)) & 1)


def kernel(days, months, out, n):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        # civil_from_days
        z = types.int64(days[i]) + 719468
        era = z // 146097
        doe = z - era * 146097
        yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
        doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
        mp = (5 * doy + 2) // 153
        d = doy - (153 * mp + 2) // 5 + 1
        m = mp + 3 if mp < 10 else mp - 9
        y = yoe + era * 400 + (1 if m <= 2 else 0)

        # Target month and day.
        target = y * 12 + (m - 1) + months[i]
        ty = target // 12
        tm = target - ty * 12 + 1
        td = min(d, _days_in_month(ty, tm))

        # days_from_civil
        yy = ty - (1 if tm <= 2 else 0)
        era2 = yy // 400
        yoe2 = yy - era2 * 400
        doy2 = (153 * (tm - 3 if tm > 2 else tm + 9) + 2) // 5 + td - 1
        doe2 = yoe2 * 365 + yoe2 // 4 - yoe2 // 100 + doy2
        out[i] = era2 * 146097 + doe2 - 719468
        i += stride


_kernel = cuda.jit(kernel)


def run(days, months, out, n):
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    blocks = max(1, min((n + THREADS - 1) // THREADS, sms * BLOCKS_PER_SM))
    _kernel[blocks, THREADS](days, months, out, n)
