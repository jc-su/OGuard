import torch
import triton
import triton.language as tl


@triton.jit
def kernel(days, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    d = tl.load(days + offs, mask=mask, other=0)
    # Civil-from-days; every input is >= -719162, so z is positive.
    z = d + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day_of_month = doy - (153 * mp + 2) // 5 + 1
    tl.store(out + offs, d - (day_of_month - 1), mask=mask)


TYPES = {"days": "*i32", "out": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(days, out, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](days, out, n, **CONSTANTS, num_warps=NUM_WARPS)
