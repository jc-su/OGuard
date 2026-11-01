import triton
import triton.language as tl


@triton.jit
def kernel(a, valid, b, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    byte = tl.load(valid + (offs >> 3), mask=mask, other=0)
    is_valid = ((byte >> (offs & 7).to(tl.uint8)) & 1) != 0
    av = tl.load(a + offs, mask=mask, other=0.0)
    bv = tl.load(b + offs, mask=mask, other=0.0)
    tl.store(out + offs, tl.where(is_valid, av, bv), mask=mask)


TYPES = {"a": "*fp64", "valid": "*u8", "b": "*fp64", "out": "*fp64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(a, valid, b, out, n):
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](a, valid, b, out, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
