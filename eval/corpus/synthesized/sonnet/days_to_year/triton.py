import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(days, year, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    days_val = tl.load(days + offs, mask=mask, other=0)

    # Howard Hinnant's civil_from_days, restricted to the year field.
    # All divisions here are C-style truncating division (Triton's `//`
    # on signed integers truncates toward zero, matching this algorithm's
    # design), and every quantity fits comfortably in int32 for years in
    # [1, 9999].
    z = days_val + 719468
    era = tl.where(z >= 0, z, z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = mp + tl.where(mp < 10, 3, -9)
    year_val = y + tl.where(m <= 2, 1, 0)

    tl.store(year + offs, year_val, mask=mask)


TYPES = {
    "days": "*i32",
    "year": "*i32",
    "n": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE}
NUM_WARPS = 8


def run(days, year, n):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](days, year, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
