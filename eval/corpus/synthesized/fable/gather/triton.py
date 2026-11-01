"""gather: out[i] = col[sel[i]] for i in [0, m). One tile of BLOCK rows per program."""

import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4

TYPES = {"col": "*fp64", "sel": "*i32", "out": "*fp64", "m": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(col, sel, out, m: tl.int64, n: tl.int64, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = rows < m
    src = tl.load(sel + rows, mask=in_bounds, other=0)
    v = tl.load(col + src, mask=in_bounds, other=0.0)
    tl.store(out + rows, v, mask=in_bounds)


def run(col, sel, out, m, n):
    grid = (max(1, triton.cdiv(m, BLOCK)),)
    kernel[grid](col, sel, out, m, n, **CONSTANTS, num_warps=NUM_WARPS)
