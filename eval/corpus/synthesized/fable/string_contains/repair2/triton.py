"""string_contains: byte substring test with lanes over (string, position) pairs.

A program owns BLOCK strings and tests PMAX candidate start positions of each
at once as a [BLOCK, PMAX] tile, walking the positions in chunks up to the
longest string of the program. For every pattern byte q the tile loads
chars[begin + p + q]: along the position axis these are consecutive bytes of
one string, so the loads coalesce, and all lanes stay busy instead of each
lane walking its own string serially. A string that is too short for a
position, or that already matched, loads nothing.

Revision: the min/max clamp of the pattern index is gone. The byte index
into `chars` is pinned in one register (an identity PTX mov) and its mask
tests it against [0, offsets[n]); the pattern byte index is pinned the same
way and its load is masked to [0, k). Under the task's contract the mask
terms never exclude a lane.
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
def pin(x):
    """Identity through one PTX mov: the mask and the address of a gather then use the same register."""
    return tl.inline_asm_elementwise("mov.b64 $0, $1;", "=l,l", [x], dtype=tl.int64, is_pure=True, pack=1)


@triton.jit
def kernel(offsets, chars, pattern, match, n: tl.int64, k: tl.int64, BLOCK: tl.constexpr, PMAX: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    if k == 0:
        tl.store(match + i, tl.full([BLOCK], 1, tl.int8), mask=m)
    else:
        beg = tl.load(offsets + i, mask=m, other=0)
        end = tl.load(offsets + i + 1, mask=m, other=0)
        total = tl.load(offsets + n).to(tl.int64)
        length = end - beg
        n_pos = tl.maximum(tl.max(length, axis=0) - k + 1, 0)
        ps = tl.arange(0, PMAX)
        found = tl.zeros([BLOCK], tl.int1)
        kp = pin(k)  # a copy of k the compiler cannot prove positive: keeps both mask terms of the pattern load
        for p0 in range(0, n_pos, PMAX):
            p = p0 + ps
            ok = ((p[None, :] + k) <= length[:, None]) & (found == 0)[:, None]
            eq = ok
            base = beg[:, None].to(tl.int64) + p[None, :].to(tl.int64)
            q = tl.cast(0, tl.int64)
            while q < k:
                idx = pin(base + q)
                c = tl.load(chars + idx, mask=ok & (idx >= 0) & (idx < total), other=0)
                qp = pin(q)
                pat = tl.load(pattern + qp, mask=(qp >= 0) & (qp < kp), other=0)
                eq = eq & (c == pat)
                q = q + 1
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
