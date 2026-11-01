import torch
import triton
import triton.language as tl


@triton.jit
def kernel(left_idx, right_idx, left, right, out, m, l, r, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < m
    li = tl.load(left_idx + offs, mask=mask, other=0).to(tl.int64)
    ri = tl.load(right_idx + offs, mask=mask, other=0).to(tl.int64)
    a = tl.load(left + li, mask=mask, other=0.0)
    b = tl.load(right + ri, mask=mask, other=0.0)
    tl.store(out + offs, a + b, mask=mask)


TYPES = {
    "left_idx": "*i32",
    "right_idx": "*i32",
    "left": "*fp64",
    "right": "*fp64",
    "out": "*fp64",
    "m": "i64",
    "l": "i64",
    "r": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(left_idx, right_idx, left, right, out, m, l, r):
    grid = (max(1, triton.cdiv(m, CONSTANTS["BLOCK"])),)
    kernel[grid](left_idx, right_idx, left, right, out, m, l, r, **CONSTANTS, num_warps=NUM_WARPS)
