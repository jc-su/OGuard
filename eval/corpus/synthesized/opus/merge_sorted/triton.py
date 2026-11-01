"""Stable merge of two sorted int64 arrays (merge path + bitonic merge).

Each program writes the BLOCK outputs [k0, k1). The merge path gives the split
of an output diagonal k, i = #{elements of a among the first k outputs}: the
smallest i with a[i] > b[k - i - 1] (so an element of a goes before an equal
element of b), a predicate that is monotone in i. Both splits of the program,
at k0 and k1, are found together by a PROBES-ary search that samples PROBES
positions on each diagonal per step.

The tile then holds a[i0:i1] ascending followed by b[j0:j1] descending, a
bitonic sequence of i1 - i0 + j1 - j0 = k1 - k0 values (the last tile is padded
in the middle with the largest int64), and a bitonic merge network of log2(BLOCK)
compare-exchange stages sorts it in registers. The merged tile is stored with
contiguous, coalesced writes. Equal keys are indistinguishable in the output,
so the network does not need to keep a before b.
"""

import torch
import triton
import triton.language as tl

INT64_MAX = tl.constexpr(2**63 - 1)


@triton.jit
def _merge_splits(a, b, m, p, k0, k1, PROBES: tl.constexpr):
    """#{elements of a among the first k outputs}, for k = k0 and k = k1."""
    which = tl.arange(0, 2)
    k = tl.where(which == 0, k0, k1)
    # Invariant: the split lies in [lo, lo + span].
    lo = tl.maximum(k - p, 0)
    span = tl.minimum(k, m) - lo
    probe = tl.arange(0, PROBES).to(tl.int64)
    while tl.max(span, axis=0) > 0:
        step = (span + PROBES - 1) // PROBES
        i = lo[:, None] + (probe[None, :] + 1) * step[:, None] - 1
        sampled = (i < (lo + span)[:, None]) & (span[:, None] > 0)
        after = tl.load(a + i, mask=sampled, other=0) > tl.load(
            b + (k[:, None] - i - 1), mask=sampled, other=0
        )
        # The sampled positions where a[i] is not yet after b form a prefix.
        before = tl.sum((sampled & ~after).to(tl.int64), axis=1)
        lo = lo + before * step
        span = tl.where(span > 0, tl.minimum(step - 1, span - before * step), 0)
    return tl.sum(tl.where(which == 0, lo, 0), axis=0), tl.sum(tl.where(which == 1, lo, 0), axis=0)


@triton.jit
def _xor(x, y):
    return x ^ y


@triton.jit
def _bitonic_merge(x, LOG: tl.constexpr):
    """Sort a bitonic sequence of 2**LOG int64 values into ascending order."""
    # Dimension d of the [2] * LOG view pairs the elements 2**(LOG - 1 - d) apart.
    y = tl.reshape(x, [2] * LOG)
    for d in tl.static_range(LOG):
        partner = y ^ tl.reduce(y, d, _xor, keep_dims=True)
        upper = tl.reshape(tl.arange(0, 2), [1] * d + [2] + [1] * (LOG - 1 - d)) == 1
        # The lower element keeps the minimum of the pair, the upper the maximum.
        y = tl.where((y > partner) != upper, partner, y)
    return tl.reshape(y, [2**LOG])


@triton.jit
def kernel(a, b, out, m, p, BLOCK: tl.constexpr, PROBES: tl.constexpr, LOG_BLOCK: tl.constexpr):
    tl.static_assert(BLOCK == 2**LOG_BLOCK)
    m64 = tl.cast(m, tl.int64)
    p64 = tl.cast(p, tl.int64)
    k0 = tl.program_id(0).to(tl.int64) * BLOCK
    k1 = tl.minimum(k0 + BLOCK, m64 + p64)
    i0, i1 = _merge_splits(a, b, m64, p64, k0, k1, PROBES)
    j0 = k0 - i0
    na = (i1 - i0).to(tl.int32)
    nb = (k1 - k0 - na).to(tl.int32)
    t = tl.arange(0, BLOCK)
    from_a = t < na
    from_b = t >= BLOCK - nb
    value_a = tl.load(a + i0 + t, mask=from_a, other=INT64_MAX)
    value_b = tl.load(b + j0 + (BLOCK - 1 - t), mask=from_b, other=INT64_MAX)
    merged = _bitonic_merge(tl.where(from_b, value_b, value_a), LOG_BLOCK)
    tl.store(out + k0 + t, merged, mask=t < k1 - k0)


TYPES = {"a": "*i64", "b": "*i64", "out": "*i64", "m": "i64", "p": "i64"}
CONSTANTS = {"BLOCK": 2048, "PROBES": 8, "LOG_BLOCK": 11}
NUM_WARPS = 4


def run(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, m: int, p: int) -> None:
    grid = (triton.cdiv(m + p, CONSTANTS["BLOCK"]),)
    kernel[grid](a, b, out, m, p, num_warps=NUM_WARPS, **CONSTANTS)
