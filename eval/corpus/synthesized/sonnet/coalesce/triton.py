import torch
import triton
import triton.language as tl


@triton.jit
def kernel(a, valid, b, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    byte = tl.load(valid + (i >> 3), mask=m, other=0).to(tl.int32)
    ok = ((byte >> (i & 7).to(tl.int32)) & 1) != 0
    av = tl.load(a + i, mask=m & ok, other=0.0)
    bv = tl.load(b + i, mask=m & ~ok, other=0.0)
    tl.store(out + i, tl.where(ok, av, bv), mask=m)


TYPES = {"a": "*fp64", "valid": "*u8", "b": "*fp64", "out": "*fp64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(a, valid, b, out, n):
    if n == 0:
        return
    kernel[(triton.cdiv(n, CONSTANTS["BLOCK"]),)](
        a, valid, b, out, n, **CONSTANTS, num_warps=NUM_WARPS
    )
