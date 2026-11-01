"""Row-wise byte equality of two Arrow string columns.

match[i] = 1 when string i of (a_offsets, a_chars) equals string i of
(b_offsets, b_chars) byte for byte, and 0 otherwise.

Each program compares ROWS rows. It loads their offsets, rules out the rows
whose lengths differ, and then compares the remaining rows CHUNK bytes at a
time as a [ROWS, CHUNK] tile. A row stops loading bytes at its first
mismatching chunk.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    a_offsets,
    a_chars,
    b_offsets,
    b_chars,
    match,
    n,
    ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    in_range = rows < n
    a_start = tl.load(a_offsets + rows, mask=in_range, other=0)
    a_end = tl.load(a_offsets + rows + 1, mask=in_range, other=0)
    b_start = tl.load(b_offsets + rows, mask=in_range, other=0)
    b_end = tl.load(b_offsets + rows + 1, mask=in_range, other=0)
    length = a_end - a_start
    equal = in_range & (length == b_end - b_start)
    max_length = tl.max(tl.where(equal, length, 0), axis=0)
    cols = tl.arange(0, CHUNK)
    for chunk in range(0, max_length, CHUNK):
        pos = chunk + cols[None, :]
        live = equal[:, None] & (pos < length[:, None])
        a_byte = tl.load(a_chars + (a_start[:, None] + pos), mask=live, other=0)
        b_byte = tl.load(b_chars + (b_start[:, None] + pos), mask=live, other=0)
        mismatches = tl.sum((a_byte != b_byte).to(tl.int32), axis=1)
        equal = equal & (mismatches == 0)
    tl.store(match + rows, equal.to(tl.int8), mask=in_range)


TYPES: dict[str, str] = {
    "a_offsets": "*i32",
    "a_chars": "*u8",
    "b_offsets": "*i32",
    "b_chars": "*u8",
    "match": "*i8",
    "n": "i64",
}
CONSTANTS: dict[str, int] = {"ROWS": 128, "CHUNK": 8}
NUM_WARPS: int = 4


def run(
    a_offsets: torch.Tensor,
    a_chars: torch.Tensor,
    b_offsets: torch.Tensor,
    b_chars: torch.Tensor,
    match: torch.Tensor,
    n: int,
) -> None:
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](a_offsets, a_chars, b_offsets, b_chars, match, n, **CONSTANTS, num_warps=NUM_WARPS)
