import torch
import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, out, n, BLOCK: tl.constexpr):
    total = tl.load(offsets + n).to(tl.int64)
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < total
    c = tl.load(chars + i, mask=m, other=0)
    lower = (c >= 97) & (c <= 122)
    tl.store(out + i, tl.where(lower, c - 32, c).to(tl.uint8), mask=m)


TYPES = {"offsets": "*i32", "chars": "*u8", "out": "*u8", "n": "i64"}
CONSTANTS = {"BLOCK": 2048}
NUM_WARPS = 4


def run(offsets, chars, out, n):
    total = chars.numel()
    if total == 0:
        return
    kernel[(triton.cdiv(total, CONSTANTS["BLOCK"]),)](
        offsets, chars, out, n, **CONSTANTS, num_warps=NUM_WARPS
    )
