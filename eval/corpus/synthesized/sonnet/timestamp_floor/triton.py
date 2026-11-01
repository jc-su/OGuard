import torch
import triton
import triton.language as tl


@triton.jit
def kernel(ts, out, n, unit, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    inside = i < n
    t = tl.load(ts + i, mask=inside, other=0)
    r = t % unit
    r = tl.where(r < 0, r + unit, r)
    tl.store(out + i, t - r, mask=inside)


TYPES = {"ts": "*i64", "out": "*i64", "n": "i64", "unit": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(ts, out, n, unit):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](ts, out, n, unit, **CONSTANTS, num_warps=NUM_WARPS)
