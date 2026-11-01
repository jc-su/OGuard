"""Fixed-width prefixes of a string column, in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n", "k"])
def kernel(offsets, chars, out, n, k, BLOCK: tl.constexpr):
    # Each program writes the k output bytes of BLOCK // k consecutive rows,
    # one output byte per lane, so the stores cover a contiguous range of out.
    width = k.to(tl.int32)
    rows_per_program = BLOCK // width
    first_row = tl.program_id(0).to(tl.int64) * rows_per_program
    lane = tl.arange(0, BLOCK)
    local_row = lane // width
    col = lane - local_row * width
    row = first_row + local_row
    in_range = (local_row < rows_per_program) & (row < n)
    begin = tl.load(offsets + row, mask=in_range, other=0)
    end = tl.load(offsets + row + 1, mask=in_range, other=0)
    copy = in_range & (col < end - begin)
    byte = tl.load(chars + begin + col, mask=copy, other=0)
    tl.store(out + first_row * width + lane, byte, mask=in_range)


TYPES: dict[str, str] = {
    "offsets": "*i32",
    "chars": "*u8",
    "out": "*u8",
    "n": "i64",
    "k": "i64",
}
CONSTANTS: dict[str, int] = {"BLOCK": 1024}
NUM_WARPS: int = 4


def run(
    offsets: torch.Tensor,
    chars: torch.Tensor,
    out: torch.Tensor,
    n: int,
    k: int,
) -> None:
    if not 1 <= k <= 64:
        raise ValueError(f"k must be in [1, 64], got {k}")
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"] // k),)
    kernel[grid](offsets, chars, out, n, k, **CONSTANTS, num_warps=NUM_WARPS)
