import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(keys + i, mask=m, other=2147483647)
    s = tl.sort(v, dim=0)
    tl.store(out + i, s, mask=m)


TYPES = {"keys": "*i32", "out": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(keys, out, n):
    if n == 0:
        return
    kernel[(triton.cdiv(n, CONSTANTS["BLOCK"]),)](
        keys, out, n, **CONSTANTS, num_warps=NUM_WARPS
    )
