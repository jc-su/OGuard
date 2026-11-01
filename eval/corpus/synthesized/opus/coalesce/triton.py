"""Coalesce a nullable float64 column with a second column.

out[i] = a[i] when bit i % 8 of valid[i / 8] is set (Arrow validity bitmap,
least significant bit first), and b[i] otherwise.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(a, valid, b, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    byte = tl.load(valid + (offs >> 3), mask=in_range, other=0)
    bit = (byte >> (offs & 7).to(tl.uint8)) & 1
    is_valid = bit != 0
    is_null = bit == 0
    # Each column is read only at the rows that take their value from it, so a
    # memory sector whose rows all come from one column is not read from the other.
    a_val = tl.load(a + offs, mask=in_range & is_valid)
    b_val = tl.load(b + offs, mask=in_range & is_null)
    tl.store(out + offs, tl.where(is_valid, a_val, b_val), mask=in_range)


TYPES: dict[str, str] = {
    "a": "*fp64",
    "valid": "*u8",
    "b": "*fp64",
    "out": "*fp64",
    "n": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 8


def run(a: torch.Tensor, valid: torch.Tensor, b: torch.Tensor, out: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](a, valid, b, out, n, **CONSTANTS, num_warps=NUM_WARPS)
