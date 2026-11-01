"""Stream compaction of the kept rows of an int64 column, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n"])
def kernel(values, keep, out, count, n, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    kept = tl.load(keep + rows, mask=rows < n, other=0) != 0
    flags = kept.to(tl.int32)
    # One atomic per program reserves a contiguous range of out for its kept rows.
    base = tl.atomic_add(count, tl.sum(flags, axis=0))
    slot = base + tl.cumsum(flags, axis=0) - 1
    tl.store(out + slot, tl.load(values + rows, mask=kept), mask=kept)


TYPES: dict[str, str] = {
    "values": "*i64",
    "keep": "*i8",
    "out": "*i64",
    "count": "*i32",
    "n": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(
    values: torch.Tensor,
    keep: torch.Tensor,
    out: torch.Tensor,
    count: torch.Tensor,
    n: int,
) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, keep, out, count, n, **CONSTANTS, num_warps=NUM_WARPS)
