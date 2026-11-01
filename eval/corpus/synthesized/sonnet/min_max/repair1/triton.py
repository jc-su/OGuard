import triton
import triton.language as tl

BLOCK_SIZE = 256


@triton.jit
def kernel(values, result, n, BLOCK_SIZE: tl.constexpr):
    # Compute the flat offset in 64-bit arithmetic. program_id and arange
    # are 32-bit by default, and for an adversarial (oversized) grid
    # `pid * BLOCK_SIZE` can overflow a 32-bit accumulator; on overflow the
    # wrapped value can go negative and still pass a 32-bit `offs < n`
    # check, producing a real out-of-bounds address. Casting to int64
    # before the multiply keeps `offs` exact (and non-negative) for any
    # grid, since gridDim.x * BLOCK_SIZE fits comfortably in 64 bits.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
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
