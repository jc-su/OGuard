import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(codes, dictionary, out, n, d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    code_vals = tl.load(codes + offs, mask=mask, other=0)
    vals = tl.load(dictionary + code_vals, mask=mask, other=0.0)
    tl.store(out + offs, vals, mask=mask)


TYPES = {
    "codes": "*i32",
    "dictionary": "*fp64",
    "out": "*fp64",
    "n": "i64",
    "d": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE}
NUM_WARPS = 8


def run(codes, dictionary, out, n, d):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](codes, dictionary, out, n, d, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
