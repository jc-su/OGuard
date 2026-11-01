"""Per-probe match counts against a sorted int64 build column, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["b", "p"])
def kernel(build, probe, counts, b, p, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < p
    key = tl.load(probe + rows, mask=in_range, other=0)
    # Branch-free binary search for the lower bound (first element >= key) and
    # the upper bound (first element > key). Every lane halves the same length,
    # so all lanes run the same ceil(log2(b)) iterations. The invariant
    # lower + length <= b (and the same for upper) keeps every load in bounds.
    lower = tl.zeros([BLOCK], dtype=tl.int64)
    upper = tl.zeros([BLOCK], dtype=tl.int64)
    length = b.to(tl.int64)
    while length > 1:
        half = length // 2
        below = tl.load(build + lower + half, mask=in_range, other=0) < key
        lower = tl.where(below, lower + half, lower)
        not_above = tl.load(build + upper + half, mask=in_range, other=0) <= key
        upper = tl.where(not_above, upper + half, upper)
        length -= half
    nonempty = in_range & (b > 0)
    last_lower = tl.load(build + lower, mask=nonempty, other=0)
    last_upper = tl.load(build + upper, mask=nonempty, other=0)
    lower += (nonempty & (last_lower < key)).to(tl.int64)
    upper += (nonempty & (last_upper <= key)).to(tl.int64)
    tl.store(counts + rows, (upper - lower).to(tl.int32), mask=in_range)


TYPES: dict[str, str] = {
    "build": "*i64",
    "probe": "*i64",
    "counts": "*i32",
    "b": "i64",
    "p": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 128}
NUM_WARPS: int = 4


def run(
    build: torch.Tensor,
    probe: torch.Tensor,
    counts: torch.Tensor,
    b: int,
    p: int,
) -> None:
    grid = (triton.cdiv(p, CONSTANTS["BLOCK"]),)
    kernel[grid](build, probe, counts, b, p, **CONSTANTS, num_warps=NUM_WARPS)
