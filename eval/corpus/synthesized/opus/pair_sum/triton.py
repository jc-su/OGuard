"""Sum of gathered float64 values from two columns, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["m", "l", "r"])
def kernel(left_idx, right_idx, left, right, out, m, l, r, BLOCK: tl.constexpr):  # noqa: E741
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < m
    a = tl.load(left + tl.load(left_idx + rows, mask=in_range, other=0), mask=in_range)
    b = tl.load(right + tl.load(right_idx + rows, mask=in_range, other=0), mask=in_range)
    tl.store(out + rows, a + b, mask=in_range)


TYPES: dict[str, str] = {
    "left_idx": "*i32",
    "right_idx": "*i32",
    "left": "*fp64",
    "right": "*fp64",
    "out": "*fp64",
    "m": "i64",
    "l": "i64",
    "r": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(
    left_idx: torch.Tensor,
    right_idx: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
    m: int,
    l: int,  # noqa: E741
    r: int,
) -> None:
    grid = (triton.cdiv(m, CONSTANTS["BLOCK"]),)
    kernel[grid](left_idx, right_idx, left, right, out, m, l, r, **CONSTANTS, num_warps=NUM_WARPS)
