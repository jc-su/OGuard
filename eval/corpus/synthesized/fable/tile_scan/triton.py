import triton
import triton.language as tl


@triton.jit
def kernel(values, out, totals, n, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * TILE + tl.arange(0, TILE)
    mask = offs < n
    v = tl.load(values + offs, mask=mask, other=0)
    inclusive = tl.cumsum(v, axis=0)
    tl.store(out + offs, inclusive - v, mask=mask)
    tl.store(totals + pid, tl.sum(v, axis=0))


TYPES = {"values": "*i32", "out": "*i32", "totals": "*i32", "n": "i64"}
CONSTANTS = {"TILE": 1024}
NUM_WARPS = 4


def run(values, out, totals, n):
    grid = (triton.cdiv(n, CONSTANTS["TILE"]),)
    kernel[grid](values, out, totals, n, TILE=CONSTANTS["TILE"], num_warps=NUM_WARPS)
