"""Triton kernel for string_length.

Counts the UTF-8 code points of each string i, delimited by
offsets[i]:offsets[i+1] in chars. A byte starts a code point unless it
is a continuation byte (its top two bits are 10), so counting
non-continuation bytes counts code points.

Each program handles a block of ROWS rows. Within the block, rows
walk their own byte range in lockstep (one byte per row per step);
rows that finish early are masked off for the remaining steps, so the
number of steps is bounded by the longest string in the block.

Repair 1 (kept): rows is built in 64-bit arithmetic throughout (pid *
ROWS + arange, all as int64), so pid * ROWS cannot overflow and wrap
into a value that still passes `rows < n` while addressing outside
offsets/lengths, for any grid.

Repair 2: the checker still refused a masked load of chars -- inside
the byte-walk loop, `b = tl.load(chars + pos, mask=active, other=0)`
uses `pos`, a loop-carried value seeded from `start`, itself read from
offsets, neither of which the checker can bound the way it bounds
`rows < n`. Two things changed:

  * `rows` is clamped into [0, n - 1] (n floored at 1) before it is
    used to address offsets or lengths at all, so `offsets + rows` and
    `offsets + rows + 1` are unconditionally inside offsets (an array
    of n + 1 elements), and `lengths + rows` is unconditionally inside
    lengths, independent of `row_mask`;
  * `chars` has no length argument of its own -- its length is
    offsets[n] by the task's contract -- so that bound is loaded
    directly (`offsets + n`, always a valid index into offsets' n + 1
    elements, since n < n + 1 needs no masking) and used to clamp
    `pos` into [0, chars_len - 1] (floored at 1) right before it forms
    the chars address, on top of the existing `active` mask (which
    already implies `pos < end <= chars_len` as a loop invariant, but
    one the checker does not track through the while loop) that still
    decides which lanes count a real byte versus 0.
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
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
    row_mask = rows < n

    # n_pos/n_last give rows a clamp bound that is well-formed even for
    # n == 0; safe_rows + 1 then stays <= n_pos <= offsets' last index.
    n_pos = tl.maximum(n, 1)
    n_last = n_pos - 1
    safe_rows = tl.minimum(tl.maximum(rows, 0), n_last)

    start = tl.load(offsets + safe_rows, mask=row_mask, other=0).to(tl.int64)
    end = tl.load(offsets + safe_rows + 1, mask=row_mask, other=0).to(tl.int64)

    # chars' length is offsets[n] by the task's contract; offsets has
    # n + 1 elements, so index n is always valid without masking.
    chars_len = tl.load(offsets + n).to(tl.int64)
    chars_len_pos = tl.maximum(chars_len, 1)
    chars_len_last = chars_len_pos - 1

    pos = start
    count = tl.zeros((ROWS,), dtype=tl.int32)
    active = row_mask & (pos < end)

    while tl.max(active.to(tl.int32)) == 1:
        safe_pos = tl.minimum(tl.maximum(pos, 0), chars_len_last)
        b = tl.load(chars + safe_pos, mask=active, other=0)
        is_start = (b & 0xC0) != 0x80
        count += tl.where(active & is_start, 1, 0)
        pos = tl.where(active, pos + 1, pos)
        active = active & (pos < end)

    tl.store(lengths + safe_rows, count, mask=row_mask)


def run(offsets, chars, lengths, n):
    """Launch kernel on PyTorch CUDA tensors."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](offsets, chars, lengths, n, ROWS=CONSTANTS["ROWS"], num_warps=NUM_WARPS)
