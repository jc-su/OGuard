import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(values, result, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    v = tl.load(values + offs, mask=mask, other=0.0)
    vmin = tl.where(mask, v, float("inf"))
    vmax = tl.where(mask, v, float("-inf"))

    local_min = tl.min(vmin, axis=0)
    local_max = tl.max(vmax, axis=0)

    tl.atomic_min(result + 0, local_min)
    tl.atomic_max(result + 1, local_max)


TYPES = {
    "values": "*fp64",
    "result": "*fp64",
    "n": "i64",
}
CONSTANTS = {"BLOCK_SIZE": BLOCK_SIZE}
NUM_WARPS = 8


def run(values, result, n):
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](values, result, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
