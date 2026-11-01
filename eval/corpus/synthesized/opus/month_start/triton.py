"""First day of the month of a date32 column, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n"])
def kernel(days, out, n, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    day = tl.load(days + rows, mask=in_range, other=0)
    # Civil-from-days conversion (H. Hinnant) with years that start on
    # March 1, counted from 0000-03-01. For days in [-719162, 2932896] the
    # shifted day z is positive, so every division below is on nonnegative
    # values and truncation equals floor division.
    z = day + 719468
    era = z // 146097
    day_of_era = z - era * 146097
    year_of_era = (day_of_era - day_of_era // 1460 + day_of_era // 36524 - day_of_era // 146096) // 365
    day_of_year = day_of_era - (365 * year_of_era + year_of_era // 4 - year_of_era // 100)
    month = (5 * day_of_year + 2) // 153
    day_of_month = day_of_year - (153 * month + 2) // 5
    tl.store(out + rows, day - day_of_month, mask=in_range)


TYPES: dict[str, str] = {"days": "*i32", "out": "*i32", "n": "i64"}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(days: torch.Tensor, out: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](days, out, n, **CONSTANTS, num_warps=NUM_WARPS)
