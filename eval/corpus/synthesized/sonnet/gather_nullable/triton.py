import torch
import triton
import triton.language as tl


@triton.jit
def kernel(col, idx, out, m, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = i < m
    j = tl.load(idx + i, mask=inb, other=-1).to(tl.int64)
    v = tl.load(col + j, mask=inb & (j >= 0), other=float("nan"))
    tl.store(out + i, v, mask=inb)


TYPES = {"col": "*fp64", "idx": "*i32", "out": "*fp64", "m": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(col, idx, out, m, n):
    if m == 0:
        return
    kernel[(triton.cdiv(m, CONSTANTS["BLOCK"]),)](
        col, idx, out, m, n, **CONSTANTS, num_warps=NUM_WARPS
    )
