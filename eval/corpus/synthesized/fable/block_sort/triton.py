import triton
import triton.language as tl


@triton.jit
def kernel(keys, out, n, TILE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * TILE + tl.arange(0, TILE)
    mask = offs < n
    v = tl.load(keys + offs, mask=mask, other=2147483647)
    tl.store(out + offs, tl.sort(v, dim=0), mask=mask)


TYPES = {"keys": "*i32", "out": "*i32", "n": "i64"}
CONSTANTS = {"TILE": 1024}
NUM_WARPS = 4


def run(keys, out, n):
    grid = (triton.cdiv(n, CONSTANTS["TILE"]),)
    kernel[grid](keys, out, n, TILE=CONSTANTS["TILE"], num_warps=NUM_WARPS)
