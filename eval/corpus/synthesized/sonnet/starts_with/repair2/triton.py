"""Triton kernel for starts_with.

match[i] = 1 when string i (offsets[i]:offsets[i+1] in chars) starts
with the k-byte pattern prefix, else 0. The comparison is byte-wise,
not code-point aware.

Each program handles a block of ROWS rows. Within the block, rows
compare their own byte against the pattern in lockstep; a row that
already mismatched, or has compared all k bytes, is masked off for
the remaining steps.

Repair 1 (kept): rows is built in 64-bit arithmetic throughout (pid *
ROWS + arange, all as int64), so pid * ROWS cannot overflow and wrap
into a value that still passes `rows < n` while addressing outside
offsets/match, for any grid.

Repair 2: the checker still refused a masked load of chars -- inside
the byte-compare loop, `c = tl.load(chars + start + j, mask=active,
other=0)` uses `start`, a value read from offsets, and `j`, a
loop-carried counter, neither of which the checker can bound the way
it bounds `rows < n`. Three things changed:

  * `rows` is clamped into [0, n - 1] (n floored at 1) before it is
    used to address offsets or match at all, so `offsets + rows` and
    `offsets + rows + 1` are unconditionally inside offsets (an array
    of n + 1 elements), and `match + rows` is unconditionally inside
    match, independent of `row_mask`;
  * `chars` has no length argument of its own -- its length is
    offsets[n] by the task's contract -- so that bound is loaded
    directly (`offsets + n`, always a valid index into offsets' n + 1
    elements, since n < n + 1 needs no masking) and used to clamp
    `start + j` into [0, chars_len - 1] (floored at 1) right before it
    forms the chars address, on top of the existing `active` mask
    that still decides which lanes compare a real byte versus 0;
  * `j` is clamped into [0, k - 1] (k floored at 1) before it forms
    the prefix address, on top of the existing `active` mask (which
    already implies `j < k`, but only as a loop invariant the checker
    does not track through the while loop).
"""

import triton
import triton.language as tl

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "prefix": "*u8",
    "match": "*i8",
    "n": "i64",
    "k": "i64",
}

CONSTANTS = {"ROWS": 128}

NUM_WARPS = 4


@triton.jit
def kernel(offsets, chars, prefix, match, n, k, ROWS: tl.constexpr):
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
    length = end - start

    ok = row_mask & (length >= k)

    # chars' length is offsets[n] by the task's contract; offsets has
    # n + 1 elements, so index n is always valid without masking.
    chars_len = tl.load(offsets + n).to(tl.int64)
    chars_len_pos = tl.maximum(chars_len, 1)
    chars_len_last = chars_len_pos - 1

    k_pos = tl.maximum(k, 1)
    k_last = k_pos - 1

    j = tl.zeros((ROWS,), dtype=tl.int64)
    active = ok & (j < k)

    while tl.max(active.to(tl.int32)) == 1:
        char_idx = tl.minimum(tl.maximum(start + j, 0), chars_len_last)
        safe_j = tl.minimum(tl.maximum(j, 0), k_last)
        c = tl.load(chars + char_idx, mask=active, other=0)
        p = tl.load(prefix + safe_j, mask=active, other=0)
        mismatch = active & (c != p)
        ok = ok & (~mismatch)
        j = tl.where(active, j + 1, j)
        active = active & (~mismatch) & (j < k)

    tl.store(match + safe_rows, ok.to(tl.int8), mask=row_mask)


def run(offsets, chars, prefix, match, n, k):
    """Launch kernel on PyTorch CUDA tensors."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](
        offsets, chars, prefix, match, n, k, ROWS=CONSTANTS["ROWS"], num_warps=NUM_WARPS
    )
