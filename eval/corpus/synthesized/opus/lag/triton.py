"""LAG window function over partitions of consecutive rows.

out[i] = values[i - k] when i - k >= first[i], and NaN otherwise, where
first[i] <= i is the first row of row i's partition and k is 1 to 1024.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, first, out, n, k, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    start = tl.load(first + offs, mask=in_range, other=0)
    src = offs - k
    val = tl.load(values + src, mask=in_range & (src >= start), other=float("nan"))
    tl.store(out + offs, val, mask=in_range)


TYPES: dict[str, str] = {
    "values": "*fp64",
    "first": "*i32",
    "out": "*fp64",
    "n": "i64",
    "k": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(values: torch.Tensor, first: torch.Tensor, out: torch.Tensor, n: int, k: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](values, first, out, n, k, **CONSTANTS, num_warps=NUM_WARPS)
