"""sorted_contains: membership of each key in a sorted distinct set.

Each lane finds j = |{t : members[t] < key}| by binary lifting (uniform steps,
no divergence), then tests members[j] == key.
"""

import torch
import triton
import triton.language as tl

BLOCK = 512
NUM_WARPS = 8
TYPES = {"keys": "*i64", "members": "*i64", "match": "*i8", "n": "i64", "s": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(keys, members, match, n: tl.int64, s: tl.int64, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    key = tl.load(keys + i, mask=m, other=0)
    # largest power of two <= s; the lifting steps then sum to at least s
    step = tl.cast(1, tl.int64)
    while step * 2 <= s:
        step = step * 2
    j = tl.zeros([BLOCK], tl.int64)
    while step > 0:
        cand = j + step
        ok = m & (cand <= s)
        v = tl.load(members + (cand - 1), mask=ok, other=9223372036854775807)
        j = tl.where(v < key, cand, j)
        step = step // 2
    hit = m & (j < s)
    v = tl.load(members + j, mask=hit, other=0)
    tl.store(match + i, (hit & (v == key)).to(tl.int8), mask=m)


def run(keys: torch.Tensor, members: torch.Tensor, match: torch.Tensor, n: int, s: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](keys, members, match, n, s, BLOCK=BLOCK, num_warps=NUM_WARPS)
