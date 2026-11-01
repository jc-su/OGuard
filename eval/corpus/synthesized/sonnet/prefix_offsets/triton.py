import torch
import triton
import triton.language as tl


@triton.jit
def kernel(lengths, offsets, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    start = pid * BLOCK
    lane = tl.arange(0, BLOCK)
    # Sum of all lengths before this block (int32, wraps like the offsets).
    carry = tl.zeros([BLOCK], dtype=tl.int32)
    for j in range(0, start, BLOCK):
        carry += tl.load(lengths + j + lane, mask=j + lane < start, other=0)
    base = tl.sum(carry, 0)
    offs = start + lane
    mask = offs < n
    x = tl.load(lengths + offs, mask=mask, other=0)
    inclusive = tl.cumsum(x, 0) + base
    tl.store(offsets + offs + 1, inclusive, mask=mask)
    tl.store(offsets + offs, base * 0, mask=(offs == 0))


TYPES = {"lengths": "*i32", "offsets": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 4096}
NUM_WARPS = 8


def run(lengths, offsets, n):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](lengths, offsets, n, **CONSTANTS, num_warps=NUM_WARPS)
