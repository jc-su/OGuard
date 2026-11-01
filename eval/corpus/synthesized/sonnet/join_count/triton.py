import torch
import triton
import triton.language as tl


@triton.jit
def kernel(build, probe, counts, b, p, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < p
    x = tl.load(probe + offs, mask=mask, other=0)
    lo = tl.zeros([BLOCK], dtype=tl.int64)
    hi = tl.zeros([BLOCK], dtype=tl.int64) + b
    for _ in range(STEPS):
        active = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(build + mid, mask=active & mask, other=0)
        go = v < x
        lo = tl.where(active & go, mid + 1, lo)
        hi = tl.where(active & (~go), mid, hi)
    lower = lo
    lo = lower
    hi = tl.zeros([BLOCK], dtype=tl.int64) + b
    for _ in range(STEPS):
        active = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(build + mid, mask=active & mask, other=0)
        go = v <= x
        lo = tl.where(active & go, mid + 1, lo)
        hi = tl.where(active & (~go), mid, hi)
    tl.store(counts + offs, (lo - lower).to(tl.int32), mask=mask)


TYPES = {"build": "*i64", "probe": "*i64", "counts": "*i32", "b": "i64", "p": "i64"}
CONSTANTS = {"BLOCK": 256, "STEPS": 32}
NUM_WARPS = 4


def run(build, probe, counts, b, p):
    assert b < 2 ** CONSTANTS["STEPS"]
    grid = (max(1, triton.cdiv(p, CONSTANTS["BLOCK"])),)
    kernel[grid](build, probe, counts, b, p, **CONSTANTS, num_warps=NUM_WARPS)
