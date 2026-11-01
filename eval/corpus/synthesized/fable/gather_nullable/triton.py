import triton
import triton.language as tl


@triton.jit
def kernel(col, idx, out, m, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < m
    row = tl.load(idx + offs, mask=mask, other=-1)
    matched = row >= 0
    v = tl.load(col + row.to(tl.int64), mask=matched, other=float("nan"))
    tl.store(out + offs, v, mask=mask)


TYPES = {"col": "*fp64", "idx": "*i32", "out": "*fp64", "m": "i64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(col, idx, out, m, n):
    grid = (triton.cdiv(m, CONSTANTS["BLOCK"]),)
    kernel[grid](col, idx, out, m, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
