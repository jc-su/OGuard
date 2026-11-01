"""Year of a date given as days since 1970-01-01 (proleptic Gregorian).

Uses the civil-from-days algorithm (H. Hinnant) in int32 arithmetic with
divisions by constants only. The day number is shifted to count from
0000-03-01, so that a leap day is the last day of its 400-, 100- and 4-year
cycle; for dates in the years 1 to 9999 the shifted value is positive, so
truncating division equals floor division. A date on or after March 1 of the
shifted year (day of year >= 306) belongs to the next calendar year.
"""

import torch
import triton
import triton.language as tl

DAYS_0000_03_01_TO_1970_01_01 = tl.constexpr(719468)
DAYS_PER_ERA = tl.constexpr(146097)  # 400 Gregorian years


@triton.jit
def kernel(days, year, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    z = tl.load(days + offs, mask=in_range, other=0) + DAYS_0000_03_01_TO_1970_01_01
    era = z // DAYS_PER_ERA
    doe = z - era * DAYS_PER_ERA
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    y = era * 400 + yoe + (doy >= 306).to(tl.int32)
    tl.store(year + offs, y, mask=in_range)


TYPES = {"days": "*i32", "year": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 8


def run(days: torch.Tensor, year: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](days, year, n, num_warps=NUM_WARPS, **CONSTANTS)
