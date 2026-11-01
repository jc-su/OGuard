"""Gather float64 rows by int32 row ids, with NaN for the id -1.

out[i] = col[idx[i]], and NaN where idx[i] is -1 (the unmatched rows of a
left outer join). Every value of idx is -1 or a row id of col.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(col, idx, out, m, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < m
    row = tl.load(idx + offs, mask=in_range, other=-1)
    val = tl.load(col + row, mask=row >= 0, other=float("nan"))
    tl.store(out + offs, val, mask=in_range)


TYPES: dict[str, str] = {
    "col": "*fp64",
    "idx": "*i32",
    "out": "*fp64",
    "m": "i64",
    "n": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(col: torch.Tensor, idx: torch.Tensor, out: torch.Tensor, m: int, n: int) -> None:
    grid = (triton.cdiv(m, CONSTANTS["BLOCK"]),)
    kernel[grid](col, idx, out, m, n, **CONSTANTS, num_warps=NUM_WARPS)
