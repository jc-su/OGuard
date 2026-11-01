"""add_months: out[i] = days[i] plus months[i] months, keeping the day of the month or
clamping it to the last day of the target month.

Dates convert with Howard Hinnant's civil_from_days / days_from_civil algorithms on
400-year eras. Every date is in the years 1 to 9999, so all the shifted day and
year counts are nonnegative and truncating division is exact floor division. The
month arithmetic counts months from year 0: target = y * 12 + (m - 1) + months.
"""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {
    "days": "*i32",
    "months": "*i32",
    "out": "*i32",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(days, months, out, n, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    # Out-of-range lanes use day 0 (1970-01-01) and month 0, a valid date.
    z = tl.load(days + rows, mask=in_range, other=0) + 719468
    add = tl.load(months + rows, mask=in_range, other=0)

    # civil_from_days
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = tl.where(mp < 10, mp + 3, mp - 9)
    y = yoe + era * 400 + (m <= 2).to(tl.int32)

    # Target month and day.
    target = y * 12 + (m - 1) + add
    ty = target // 12
    tm = target - ty * 12 + 1
    leap = ((ty % 4 == 0) & (ty % 100 != 0)) | (ty % 400 == 0)
    dim = tl.where(tm == 2, tl.where(leap, 29, 28), 30 + ((tm + (tm >> 3)) & 1))
    td = tl.minimum(d, dim)

    # days_from_civil
    yy = ty - (tm <= 2).to(tl.int32)
    era2 = yy // 400
    yoe2 = yy - era2 * 400
    doy2 = (153 * tl.where(tm > 2, tm - 3, tm + 9) + 2) // 5 + td - 1
    doe2 = yoe2 * 365 + yoe2 // 4 - yoe2 // 100 + doy2
    tl.store(out + rows, era2 * 146097 + doe2 - 719468, mask=in_range)


def run(days, months, out, n):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](days, months, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
