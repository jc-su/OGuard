"""string_contains: byte substring test with lanes over (string, position) pairs.

A program owns BLOCK strings and tests PMAX candidate start positions of each
at once as a [BLOCK, PMAX] tile, walking the positions in chunks up to the
longest string of the program. For every pattern byte q the tile loads
chars[begin + p + q]: along the position axis these are consecutive bytes of
one string, so the loads coalesce, and all lanes stay busy instead of each
lane walking its own string serially. A string that is too short for a
position, or that already matched, loads nothing.
"""

import torch
import triton
import triton.language as tl

BLOCK = 64
PMAX = 8
NUM_WARPS = 4
TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "pattern": "*u8",
    "match": "*i8",
    "n": "i64",
    "k": "i64",
}
CONSTANTS = {"BLOCK": BLOCK, "PMAX": PMAX}


@triton.jit
def kernel(offsets, chars, pattern, match, n: tl.int64, k: tl.int64, BLOCK: tl.constexpr, PMAX: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    if k == 0:
        tl.store(match + i, tl.full([BLOCK], 1, tl.int8), mask=m)
    else:
        beg = tl.load(offsets + i, mask=m, other=0)
        end = tl.load(offsets + i + 1, mask=m, other=0)
        length = end - beg
        n_pos = tl.maximum(tl.max(length, axis=0) - k + 1, 0)
        ps = tl.arange(0, PMAX)
        found = tl.zeros([BLOCK], tl.int1)
        for p0 in range(0, n_pos, PMAX):
            p = p0 + ps
            ok = ((p[None, :] + k) <= length[:, None]) & (found == 0)[:, None]
            eq = ok
            base = beg[:, None] + p[None, :]
            for q in range(0, k):
                c = tl.load(chars + (base + q), mask=ok, other=0)
                pat = tl.load(pattern + q)
                eq = eq & (c == pat)
            found = found | (tl.max(eq.to(tl.int32), axis=1) > 0)
        tl.store(match + i, found.to(tl.int8), mask=m)


def run(
    offsets: torch.Tensor,
    chars: torch.Tensor,
    pattern: torch.Tensor,
    match: torch.Tensor,
    n: int,
    k: int,
) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](offsets, chars, pattern, match, n, k, BLOCK=BLOCK, PMAX=PMAX, num_warps=NUM_WARPS)
