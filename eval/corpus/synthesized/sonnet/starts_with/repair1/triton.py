"""Triton kernel for starts_with.

match[i] = 1 when string i (offsets[i]:offsets[i+1] in chars) starts
with the k-byte pattern prefix, else 0. The comparison is byte-wise,
not code-point aware.

Each program handles a block of ROWS rows. Within the block, rows
compare their own byte against the pattern in lockstep; a row that
already mismatched, or has compared all k bytes, is masked off for
the remaining steps.

Repair: rows was computed as pid * ROWS + arange in 32-bit
arithmetic. Since the checker must prove safety for any grid, pid is
not bounded by the grid this file happens to launch, so pid * ROWS
can overflow a 32-bit index and wrap into a value that still passes
`rows < n` while addressing outside offsets/match. rows is now built
in 64-bit arithmetic throughout, so no such wraparound is possible.
start, end, j and the chars/prefix indexing stay 32-bit: those values
are all bounded by offsets' own int32-representable range, which
cannot overflow regardless of the grid.
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

    start = tl.load(offsets + rows, mask=row_mask, other=0)
    end = tl.load(offsets + rows + 1, mask=row_mask, other=0)
    length = end - start

    ok = row_mask & (length >= k)

    j = tl.zeros((ROWS,), dtype=tl.int32)
    active = ok & (j < k)

    while tl.max(active.to(tl.int32)) == 1:
        c = tl.load(chars + start + j, mask=active, other=0)
        p = tl.load(prefix + j, mask=active, other=0)
        mismatch = active & (c != p)
        ok = ok & (~mismatch)
        j = tl.where(active, j + 1, j)
        active = active & (~mismatch) & (j < k)

    tl.store(match + rows, ok.to(tl.int8), mask=row_mask)


def run(offsets, chars, prefix, match, n, k):
    """Launch kernel on PyTorch CUDA tensors."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["ROWS"]),)
    kernel[grid](
        offsets, chars, prefix, match, n, k, ROWS=CONSTANTS["ROWS"], num_warps=NUM_WARPS
    )
