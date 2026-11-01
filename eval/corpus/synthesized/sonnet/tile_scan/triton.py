import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, out, totals, n, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    i = t * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(values + i, mask=m, other=0)
    inclusive = tl.cumsum(v, axis=0)
    tl.store(out + i, inclusive - v, mask=m)
    tl.store(totals + t, tl.sum(v, axis=0))


TYPES = {"values": "*i32", "out": "*i32", "totals": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(values, out, totals, n):
    if n == 0:
        return
    kernel[(triton.cdiv(n, CONSTANTS["BLOCK"]),)](
        values, out, totals, n, **CONSTANTS, num_warps=NUM_WARPS
    )
