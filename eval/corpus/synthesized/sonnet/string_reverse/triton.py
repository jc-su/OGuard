import torch
import triton
import triton.language as tl


@triton.jit
def kernel(offsets, chars, out, n, ROWS: tl.constexpr, COLS: tl.constexpr):
    # Each program reverses ROWS consecutive strings, COLS bytes at a time.
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    inb = row < n
    begin = tl.load(offsets + row, mask=inb, other=0).to(tl.int64)
    end = tl.load(offsets + row + 1, mask=inb, other=0).to(tl.int64)
    longest = tl.max(end - begin, axis=0)
    for c in range(0, longest, COLS):
        col = c + tl.arange(0, COLS)
        live = (begin[:, None] + col[None, :]) < end[:, None]
        src = end[:, None] - 1 - col[None, :]
        dst = begin[:, None] + col[None, :]
        b = tl.load(chars + src, mask=live)
        tl.store(out + dst, b, mask=live)


TYPES = {"offsets": "*i32", "chars": "*u8", "out": "*u8", "n": "i64"}
CONSTANTS = {"ROWS": 4, "COLS": 128}
NUM_WARPS = 4


def run(offsets: torch.Tensor, chars: torch.Tensor, out: torch.Tensor, n: int) -> None:
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](offsets, chars, out, n, **CONSTANTS, num_warps=NUM_WARPS)
