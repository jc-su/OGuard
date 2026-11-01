import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, first, out, n, k, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    f = tl.load(first + i, mask=m, other=0).to(tl.int64)
    src = i - k
    ok = m & (src >= f)
    v = tl.load(values + src, mask=ok, other=float("nan"))
    tl.store(out + i, v, mask=m)


TYPES = {"values": "*fp64", "first": "*i32", "out": "*fp64", "n": "i64", "k": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(values, first, out, n, k):
    if n == 0:
        return
    kernel[(triton.cdiv(n, CONSTANTS["BLOCK"]),)](
        values, first, out, n, k, **CONSTANTS, num_warps=NUM_WARPS
    )
