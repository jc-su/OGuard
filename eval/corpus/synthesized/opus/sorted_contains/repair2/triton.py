"""Membership test of int64 keys in a sorted, distinct int64 set.

One lane per key finds the last member <= key by branchless binary lifting
over `members`: starting from pos = 0, it tries the steps
bit = 2^k, 2^(k-1), ..., 1 (2^k the highest power of two <= s) and moves to
pos + bit when that member exists and is <= key. The step is the same in every
lane, so the loop count, floor(log2(s)) + 1, is uniform and each step is one
gather. The first steps touch the same few members in every lane and are
served from cache. With s = 0 no member is read and every key is a
non-member.

Bounds: every load and store has the mask 0 <= index and index < length of
the very index it uses (n for keys and match, s for members), so its bound
needs no minimum, maximum or loop invariant. The loop runs on the step, which
comes from s by bit operations, and not on a comparison with s itself, so the
compiler keeps these comparisons as they are written. A lane past n searches
for the key 0 and stores nothing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(keys, members, match, n, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = (offs >= 0) & (offs < n)
    key = tl.load(keys + offs, mask=in_range, other=0)
    # The highest power of two <= s, or 0 when s = 0.
    high = tl.cast(s, tl.int64)
    high = high | (high >> 1)
    high = high | (high >> 2)
    high = high | (high >> 4)
    high = high | (high >> 8)
    high = high | (high >> 16)
    high = high | (high >> 32)
    bit = high - (high >> 1)
    pos = tl.zeros((BLOCK,), tl.int64)  # members[pos] <= key, or pos = 0
    while bit > 0:
        probe = pos + bit
        present = (probe >= 0) & (probe < s)
        value = tl.load(members + probe, mask=present, other=0)
        pos = tl.where(present & (value <= key), probe, pos)
        bit = bit >> 1
    present = (pos >= 0) & (pos < s)
    value = tl.load(members + pos, mask=present, other=0)
    found = present & (value == key)
    tl.store(match + offs, found.to(tl.int8), mask=in_range)


TYPES = {"keys": "*i64", "members": "*i64", "match": "*i8", "n": "i64", "s": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 8


def run(keys: torch.Tensor, members: torch.Tensor, match: torch.Tensor, n: int, s: int) -> None:
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](keys, members, match, n, s, num_warps=NUM_WARPS, **CONSTANTS)
