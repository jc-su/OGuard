"""Uppercase the ASCII letters of an Arrow string column.

out has the bytes chars[0 : offsets[n]] with each byte in 'a'..'z' replaced by
its uppercase letter, so that (offsets, out) is the uppercase column. The
kernel reads the byte count offsets[n] from device memory; the launcher sizes
the grid from the length of chars, which is offsets[n] bytes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, out, n, BLOCK: tl.constexpr):
    total = tl.load(offsets + n)
    start = tl.program_id(0).to(tl.int64) * BLOCK
    offs = start + tl.arange(0, BLOCK)
    # A mask against the runtime byte count keeps the compiler from using
    # vector loads and stores, so only the block that holds the end is masked.
    if start + BLOCK <= total:
        c = tl.load(chars + offs)
        upper = tl.where((c >= 97) & (c <= 122), c - 32, c).to(tl.uint8)
        tl.store(out + offs, upper)
    else:
        in_range = offs < total
        c = tl.load(chars + offs, mask=in_range, other=0)
        upper = tl.where((c >= 97) & (c <= 122), c - 32, c).to(tl.uint8)
        tl.store(out + offs, upper, mask=in_range)


TYPES: dict[str, str] = {
    "offsets": "*i32",
    "chars": "*u8",
    "out": "*u8",
    "n": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 4096}
NUM_WARPS: int = 4


def run(offsets: torch.Tensor, chars: torch.Tensor, out: torch.Tensor, n: int) -> None:
    grid = (triton.cdiv(chars.numel(), CONSTANTS["BLOCK"]),)
    kernel[grid](offsets, chars, out, n, **CONSTANTS, num_warps=NUM_WARPS)
