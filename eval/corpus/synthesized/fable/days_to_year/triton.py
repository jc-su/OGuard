"""days_to_year: year of a proleptic Gregorian date given as days since 1970-01-01.

Uses the civil_from_days arithmetic (Hinnant): shift the epoch to 0000-03-01,
split into 400-year eras, then years within the era. The dates are in the years
1 to 9999, so the shifted day number is positive and truncating division is
exact. Divisions by constants compile to multiply-high sequences.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
TYPES = {"days": "*i32", "year": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(days, year, n: tl.int64, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    d = tl.load(days + i, mask=m, other=0)
    z = d + 719468  # days since 0000-03-01, positive for years >= 1
    era = z // 146097
    doe = z - era * 146097  # day of era, [0, 146096]
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365  # [0, 399]
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)  # day of the March-based year
    mp = (5 * doy + 2) // 153  # March-based month, [0, 11]; 10 and 11 are Jan and Feb
    y = yoe + era * 400 + (mp >= 10).to(tl.int32)
    tl.store(year + i, y, mask=m)


def run(days: torch.Tensor, year: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](days, year, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
