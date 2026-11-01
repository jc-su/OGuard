import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    total = tl.load(offsets + n)
    mask = offs < total
    c = tl.load(chars + offs, mask=mask, other=0)
    lower = (c >= 97) & (c <= 122)
    tl.store(out + offs, tl.where(lower, c - 32, c), mask=mask)


TYPES = {"offsets": "*i32", "chars": "*u8", "out": "*u8", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(offsets, chars, out, n):
    grid = (triton.cdiv(out.numel(), CONSTANTS["BLOCK"]),)
    kernel[grid](offsets, chars, out, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
