import triton
import triton.language as tl


@triton.jit
def kernel(days, months, out, n, BLOCK: tl.constexpr):
    # Each program handles BLOCK rows: civil date from days (Hinnant's
    # algorithm, all intermediate values nonnegative for years 1 to 9999),
    # month arithmetic, day clamped to the target month, and back to days.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    msk = offs < n
    z = tl.load(days + offs, mask=msk, other=0).to(tl.int64) + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = tl.where(mp < 10, mp + 3, mp - 9)
    y = yoe + era * 400 + (m <= 2).to(tl.int64)
    total = y * 12 + (m - 1) + tl.load(months + offs, mask=msk, other=0).to(tl.int64)
    ty = total // 12
    tm = total - ty * 12 + 1
    leap = ((ty % 4 == 0) & (ty % 100 != 0)) | (ty % 400 == 0)
    dim = tl.where(tm == 2, 28 + leap.to(tl.int64), 31 - ((tm - 1) % 7) % 2)
    d = tl.minimum(d, dim)
    yy = tl.where(tm <= 2, ty - 1, ty)
    era2 = yy // 400
    yoe2 = yy - era2 * 400
    mp2 = tl.where(tm > 2, tm - 3, tm + 9)
    doy2 = (153 * mp2 + 2) // 5 + d - 1
    doe2 = yoe2 * 365 + yoe2 // 4 - yoe2 // 100 + doy2
    tl.store(out + offs, (era2 * 146097 + doe2 - 719468).to(tl.int32), mask=msk)


TYPES = {"days": "*i32", "months": "*i32", "out": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(days, months, out, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](days, months, out, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
