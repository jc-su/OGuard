import triton
import triton.language as tl


@triton.jit
def kernel(values, first, out, n, k, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    start = tl.load(first + offs, mask=mask, other=0)
    src = offs - k
    inside = mask & (src >= start)
    v = tl.load(values + src, mask=inside, other=float("nan"))
    tl.store(out + offs, v, mask=mask)


TYPES = {"values": "*fp64", "first": "*i32", "out": "*fp64", "n": "i64", "k": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(values, first, out, n, k):
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, first, out, n, k, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
