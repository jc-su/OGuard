"""Bucket float64 values by sorted bounds.

out[i] is the number of bounds that are less than or equal to values[i]
(numpy.searchsorted(bounds, values, side="right")). values has no NaN and
bounds is sorted ascending.

The binary search keeps base and length with the count in
[base, base + length] and base + length <= b, and halves length each step.
length does not depend on the row, so all rows of a program run the same
ceil(log2(b)) steps, and every read is at a position below base + length.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, bounds, out, n, b, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    v = tl.load(values + offs, mask=in_range, other=0.0)
    base = tl.zeros([BLOCK], dtype=tl.int64)
    length = b
    while length > 1:
        half = length // 2
        probe = tl.load(bounds + (base + half), mask=in_range, other=0.0)
        base = tl.where(probe <= v, base + half, base)
        length -= half
    nonempty = b > 0
    last = tl.load(bounds + base, mask=in_range & nonempty, other=0.0)
    count = base + ((last <= v) & nonempty).to(tl.int64)
    tl.store(out + offs, count.to(tl.int32), mask=in_range)


TYPES: dict[str, str] = {
    "values": "*fp64",
    "bounds": "*fp64",
    "out": "*i32",
    "n": "i64",
    "b": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(values: torch.Tensor, bounds: torch.Tensor, out: torch.Tensor, n: int, b: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, bounds, out, n, b, **CONSTANTS, num_warps=NUM_WARPS)
