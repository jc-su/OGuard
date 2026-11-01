"""projection: out[i] = price[i] * (1 - discount[i]). One tile of BLOCK rows per program."""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {"price": "*fp64", "discount": "*fp64", "out": "*fp64", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(price, discount, out, n: tl.int64, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = rows < n
    p = tl.load(price + rows, mask=in_bounds, other=0.0)
    d = tl.load(discount + rows, mask=in_bounds, other=0.0)
    tl.store(out + rows, p * (1.0 - d), mask=in_bounds)


def run(price, discount, out, n):
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    kernel[grid](price, discount, out, n, **CONSTANTS, num_warps=NUM_WARPS)
