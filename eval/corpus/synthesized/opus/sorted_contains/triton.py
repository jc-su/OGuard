"""Membership test of int64 keys in a sorted, distinct int64 set.

One lane per key runs a branchless binary search over `members`. The search
interval has the same length in every lane (only its base differs), so the loop
count, ceil(log2(s)), is uniform and each step is one gather. The first steps
touch the same few members in every lane and are served from cache. With s = 0
no member is read and every key is a non-member.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, members, match, n, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    key = tl.load(keys + offs, mask=in_range, other=0)
    if s == 0:
        tl.store(match + offs, tl.zeros((BLOCK,), tl.int8), mask=in_range)
    else:
        # Invariant: the last member <= key, if any, is in [base, base + span).
        base = tl.zeros((BLOCK,), tl.int64)
        span = tl.cast(s, tl.int64)
        while span > 1:
            half = span // 2
            probe = tl.load(members + base + half, mask=in_range, other=0)
            base = tl.where(probe <= key, base + half, base)
            span = span - half
        found = tl.load(members + base, mask=in_range, other=0) == key
        tl.store(match + offs, found.to(tl.int8), mask=in_range)


TYPES = {"keys": "*i64", "members": "*i64", "match": "*i8", "n": "i64", "s": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(keys: torch.Tensor, members: torch.Tensor, match: torch.Tensor, n: int, s: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, members, match, n, s, num_warps=NUM_WARPS, **CONSTANTS)
