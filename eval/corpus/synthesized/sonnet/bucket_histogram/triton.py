import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, bounds, counts, n, b, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    v = tl.load(values + offs, mask=inb)
    # Branchless binary search for the number of bounds <= v.
    step = b * 0 + 1
    while step * 2 <= b:
        step = step * 2
    pos = tl.zeros([BLOCK], dtype=tl.int64)
    while step > 0:
        cand = pos + step
        ok = inb & (cand <= b)
        probe = tl.load(bounds + cand - 1, mask=ok)
        pos = tl.where(ok & (probe <= v), cand, pos)
        step = step >> 1
    ones = tl.full([BLOCK], 1, tl.int32)
    tl.atomic_add(counts + pos, ones, mask=inb, sem="relaxed")


TYPES = {
    "values": "*fp64",
    "bounds": "*fp64",
    "counts": "*i32",
    "n": "i64",
    "b": "i64",
}
CONSTANTS = {"BLOCK": 512}
NUM_WARPS = 4


def run(
    values: torch.Tensor,
    bounds: torch.Tensor,
    counts: torch.Tensor,
    n: int,
    b: int,
) -> None:
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, bounds, counts, n, b, **CONSTANTS, num_warps=NUM_WARPS)
