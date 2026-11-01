"""Triton kernel for string_length.

Counts the UTF-8 code points of each string i, delimited by
offsets[i]:offsets[i+1] in chars. A byte starts a code point unless it
is a continuation byte (its top two bits are 10), so counting
non-continuation bytes counts code points.

Each program handles a block of ROWS rows. Within the block, rows
walk their own byte range in lockstep (one byte per row per step);
rows that finish early are masked off for the remaining steps, so the
number of steps is bounded by the longest string in the block.
"""

import triton
import triton.language as tl

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "lengths": "*i32",
    "n": "i64",
}

CONSTANTS = {"ROWS": 128}

NUM_WARPS = 4


@triton.jit
def kernel(offsets, chars, lengths, n, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < n

    start = tl.load(offsets + rows, mask=row_mask, other=0)
    end = tl.load(offsets + rows + 1, mask=row_mask, other=0)

    pos = start
    count = tl.zeros((ROWS,), dtype=tl.int32)
    active = row_mask & (pos < end)

    while tl.max(active.to(tl.int32)) == 1:
        b = tl.load(chars + pos, mask=active, other=0)
        is_start = (b & 0xC0) != 0x80
        count += tl.where(active & is_start, 1, 0)
        pos = tl.where(active, pos + 1, pos)
        active = active & (pos < end)

    tl.store(lengths + rows, count, mask=row_mask)


def run(offsets, chars, lengths, n):
    """Launch kernel on PyTorch CUDA tensors."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](offsets, chars, lengths, n, ROWS=CONSTANTS["ROWS"], num_warps=NUM_WARPS)
