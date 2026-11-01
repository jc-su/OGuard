"""Stable merge of two sorted int64 arrays (merge path, a short sequential merge per lane).

Each program writes the BLOCK = LANES * ITEMS outputs [k0, k1), and lane l
writes the ITEMS consecutive outputs from diagonal d = k0 + l * ITEMS. The
merge path gives the split of an output diagonal k,
i = #{elements of a among the first k outputs}: the smallest i with
a[i] > b[k - i - 1] (so an element of a goes before an equal element of b), a
predicate that is monotone in i.

1. The program finds the splits i0 and i1 of k0 and k1 with a PROBES-ary
   search per diagonal: every step samples PROBES positions, and a sum over
   the samples gives the sub-range that holds the split.
2. Each lane finds the split of its diagonal by a binary search inside
   [i0, i1], which holds at most BLOCK + 1 positions, with a step count that
   is the same in every lane (bit_length(i1 - i0)), so the lanes stay
   converged.
3. Each lane merges ITEMS outputs sequentially, one element of a and one of b
   per step.

The kernel uses elementwise operations, gathers, stores and sums over the
samples, so it moves no tile through shared memory. Every load and store index
is at least 0 by construction (a maximum with 0 where it is formed from a
difference or from a sum over lanes) and has a mask that is one comparison of
that index with the length of its array.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _split(a, b, m, p, k, PROBES: tl.constexpr):
    """The split of diagonal k: #{elements of a among the first k outputs}."""
    probe = tl.arange(0, PROBES).to(tl.int64)
    # Invariant: the split lies in [lo, lo + span].
    lo = tl.maximum(k - p, 0)
    span = tl.minimum(k, m) - lo
    while span > 0:
        step = tl.maximum((span + PROBES - 1) // PROBES, 1)
        i = lo + (probe + 1) * step - 1
        sampled = i < lo + span
        index_a = i  # at least lo >= 0
        index_b = tl.maximum(k - i - 1, 0)
        value_a = tl.load(a + index_a, mask=index_a < m, other=0)
        value_b = tl.load(b + index_b, mask=index_b < p, other=0)
        # The sampled positions where a[i] is not after b[k - i - 1] form a prefix.
        before = tl.sum((sampled & (value_a <= value_b)).to(tl.int64), axis=0)
        lo = tl.maximum(lo + before * step, 0)
        span = tl.minimum(step - 1, span - before * step)
    return lo


@triton.jit
def kernel(a, b, out, m, p, LANES: tl.constexpr, ITEMS: tl.constexpr, PROBES: tl.constexpr):
    BLOCK: tl.constexpr = LANES * ITEMS
    total = m + p
    k0 = tl.program_id(0).to(tl.int64) * BLOCK
    k1 = tl.minimum(k0 + BLOCK, total)
    i0 = _split(a, b, m, p, k0, PROBES)
    i1 = _split(a, b, m, p, k1, PROBES)

    lane = tl.arange(0, LANES).to(tl.int64)
    d = tl.minimum(k0 + lane * ITEMS, k1)  # the first output of the lane
    # The split of diagonal d lies in [lo, hi].
    lo = tl.maximum(tl.maximum(d - p, 0), i0)
    hi = tl.minimum(tl.minimum(d, m), i1)
    span = i1 - i0  # an upper bound of hi - lo
    while span > 0:
        mid = (lo + hi) >> 1
        index_a = tl.maximum(mid, 0)
        index_b = tl.maximum(d - 1 - mid, 0)
        value_a = tl.load(a + index_a, mask=index_a < m, other=0)
        value_b = tl.load(b + index_b, mask=index_b < p, other=0)
        a_first = value_a <= value_b
        active = lo < hi
        lo = tl.where(active & a_first, mid + 1, lo)
        hi = tl.where(active & ~a_first, mid, hi)
        span = span >> 1
    next_a = lo
    next_b = d - lo
    for step in tl.static_range(ITEMS):
        index_a = tl.maximum(next_a, 0)
        index_b = tl.maximum(next_b, 0)
        value_a = tl.load(a + index_a, mask=index_a < m, other=0)
        value_b = tl.load(b + index_b, mask=index_b < p, other=0)
        take_a = (next_a < m) & ((next_b >= p) | (value_a <= value_b))
        position = d + step
        tl.store(out + position, tl.where(take_a, value_a, value_b), mask=position < total)
        next_a = tl.where(take_a, next_a + 1, next_a)
        next_b = tl.where(take_a, next_b, next_b + 1)


TYPES = {"a": "*i64", "b": "*i64", "out": "*i64", "m": "i64", "p": "i64"}
CONSTANTS = {"LANES": 512, "ITEMS": 4, "PROBES": 8}
NUM_WARPS = 8


def run(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, m: int, p: int) -> None:
    grid = (triton.cdiv(m + p, CONSTANTS["LANES"] * CONSTANTS["ITEMS"]),)
    kernel[grid](a, b, out, m, p, num_warps=NUM_WARPS, **CONSTANTS)
