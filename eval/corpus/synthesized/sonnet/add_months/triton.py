import torch
import triton
import triton.language as tl


@triton.jit
def kernel(days, months, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    # z counts from 0000-03-01 and is nonnegative for the years 1 to 9999, so
    # floor and truncating division agree.
    z = tl.load(days + offs, mask=mask, other=0).to(tl.int64) + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = tl.where(mp < 10, mp + 3, mp - 9)
    y = tl.where(m <= 2, y + 1, y)

    mo = tl.load(months + offs, mask=mask, other=0).to(tl.int64)
    total = y * 12 + (m - 1) + mo
    ty = total // 12
    tm = total % 12 + 1

    leap = ((ty % 4 == 0) & (ty % 100 != 0)) | (ty % 400 == 0)
    dim = tl.where(
        tm == 2,
        tl.where(leap, 29, 28),
        tl.where((tm == 4) | (tm == 6) | (tm == 9) | (tm == 11), 30, 31),
    )
    d = tl.minimum(d, dim)

    y2 = tl.where(tm <= 2, ty - 1, ty)
    era2 = y2 // 400
    yoe2 = y2 - era2 * 400
    mp2 = tl.where(tm > 2, tm - 3, tm + 9)
    doy2 = (153 * mp2 + 2) // 5 + d - 1
    doe2 = yoe2 * 365 + yoe2 // 4 - yoe2 // 100 + doy2
    res = era2 * 146097 + doe2 - 719468
    tl.store(out + offs, res.to(tl.int32), mask=mask)


TYPES = {"days": "*i32", "months": "*i32", "out": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(days, months, out, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](days, months, out, n, **CONSTANTS, num_warps=NUM_WARPS)
