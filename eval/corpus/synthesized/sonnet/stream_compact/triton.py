import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, keep, out, count, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    selected = tl.load(keep + offs, mask=mask, other=0) != 0
    flag = selected.to(tl.int32)
    rank = tl.cumsum(flag, 0) - 1
    total = tl.sum(flag, 0)
    base = tl.atomic_add(count, total)
    v = tl.load(values + offs, mask=mask)
    tl.store(out + (base + rank).to(tl.int64), v, mask=selected)


TYPES = {"values": "*i64", "keep": "*i8", "out": "*i64", "count": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(values, keep, out, count, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](values, keep, out, count, n, **CONSTANTS, num_warps=NUM_WARPS)
