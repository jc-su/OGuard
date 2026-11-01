"""radix_scatter: the scatter pass of an LSD radix sort on one 8-bit digit.

Each program ranks its BLOCK keys locally so that the 256 cursors see one
atomic add per (program, non-empty digit) instead of one per key. The
digits are handled in chunks of DCH: for a chunk, a [DCH, BLOCK] tile marks
which key has which digit, its row sums are the digit counts of the program
(one masked atomic add per digit reserves the program's segments in ``out``
and returns their starts), and its exclusive row scans are the ranks of the
keys inside their digit. A key's position is the start of its digit's
segment plus its rank, picked out of the tile by a masked column sum.

Revision: the local ranking uses comparisons, scans and sums only, instead
of a sort and a gather whose shared memory addressing the memory checker
cannot prove, and the store mask tests the position against [0, n). The
extra terms never mask a lane out under the task's contract (the cursors
hold the exclusive prefix sum of the digit counts on entry); they state the
bound of the store for the memory checker.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
DCH = 8  # digits per ranking chunk
NUM_WARPS = 8
TYPES = {"keys": "*u32", "cursors": "*i32", "out": "*i32", "n": "i64", "shift": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "DCH": DCH}


@triton.jit
def kernel(keys, cursors, out, n: tl.int64, shift: tl.int64, BLOCK: tl.constexpr, DCH: tl.constexpr):
    start = tl.program_id(0).to(tl.int64) * BLOCK
    i = start + tl.arange(0, BLOCK)
    m = i < n
    key = tl.load(keys + i, mask=m, other=0)
    digit = ((key >> shift.to(tl.uint32)) & 255).to(tl.int32)
    pos = tl.zeros([BLOCK], tl.int64)
    for d0 in tl.static_range(0, 256, DCH):
        d = d0 + tl.arange(0, DCH)
        eq = (digit[None, :] == d[:, None]) & m[None, :]
        eqi = eq.to(tl.int32)
        cnt = tl.sum(eqi, axis=1)
        seg = tl.atomic_add(cursors + d, cnt, mask=cnt > 0, sem="relaxed")
        rank = tl.cumsum(eqi, axis=1) - eqi
        pos += tl.sum(tl.where(eq, seg[:, None] + rank, 0), axis=0).to(tl.int64)
    tl.store(out + pos, i.to(tl.int32), mask=m & (pos >= 0) & (pos < n))


def run(keys: torch.Tensor, cursors: torch.Tensor, out: torch.Tensor, n: int, shift: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](keys, cursors, out, n, shift, BLOCK=BLOCK, DCH=DCH, num_warps=NUM_WARPS)
