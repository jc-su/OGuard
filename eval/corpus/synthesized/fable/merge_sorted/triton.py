"""merge_sorted: merge path merge of two sorted int64 columns.

A program owns BLOCK output positions starting at diagonal d0. It first finds
the merge path partition at d0, the number i0 of a-elements among the first d0
outputs, with a k-ary search that probes BLOCK evenly spaced positions of the
candidate interval per round (two rounds for a million rows). Then every lane
finds its own partition inside the program's window of a and b by binary
search and emits min(a[i], b[j]), a first on ties. All searches use the same
predicate a[i] <= b[d - i - 1], which is what makes the merge stable.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
STEPS = 11  # log2(BLOCK) + 1 halvings empty any interval of at most BLOCK
NUM_WARPS = 4
TYPES = {"a": "*i64", "b": "*i64", "out": "*i64", "m": "i64", "p": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "STEPS": STEPS}


@triton.jit
def kernel(a, b, out, m: tl.int64, p: tl.int64, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    tl.static_assert((1 << (STEPS - 1)) == BLOCK)
    d0 = tl.program_id(0).to(tl.int64) * BLOCK
    t = tl.arange(0, BLOCK)
    t64 = t.to(tl.int64)

    # partition at d0: the smallest i in [lo, hi] with a[i] > b[d0 - i - 1]
    lo = tl.maximum(d0 - p, 0)
    hi = tl.minimum(d0, m)
    while hi - lo > BLOCK:
        stride = (hi - lo + BLOCK - 1) // BLOCK
        pos = lo + t64 * stride
        ok = pos < hi
        av = tl.load(a + pos, mask=ok, other=0)
        bv = tl.load(b + (d0 - pos - 1), mask=ok, other=0)
        c = tl.sum((ok & (av <= bv)).to(tl.int32), axis=0).to(tl.int64)
        new_lo = tl.where(c > 0, lo + (c - 1) * stride + 1, lo)
        new_hi = tl.minimum(hi, lo + c * stride)
        lo = new_lo
        hi = new_hi
    pos = lo + t64
    ok = pos < hi
    av = tl.load(a + pos, mask=ok, other=0)
    bv = tl.load(b + (d0 - pos - 1), mask=ok, other=0)
    i0 = lo + tl.sum((ok & (av <= bv)).to(tl.int32), axis=0).to(tl.int64)
    j0 = d0 - i0

    # per lane: partition at relative diagonal t within a[i0:i0+na], b[j0:j0+nb]
    na = tl.minimum(m - i0, BLOCK).to(tl.int32)
    nb = tl.minimum(p - j0, BLOCK).to(tl.int32)
    lo_v = tl.maximum(t - nb, 0)
    hi_v = tl.minimum(t, na)
    for _ in tl.static_range(STEPS):
        active = lo_v < hi_v
        mid = (lo_v + hi_v) // 2
        av = tl.load(a + (i0 + mid), mask=active, other=0)
        bv = tl.load(b + (j0 + (t - mid - 1)), mask=active, other=0)
        take_a = av <= bv
        lo_v = tl.where(active & take_a, mid + 1, lo_v)
        hi_v = tl.where(active & (take_a == 0), mid, hi_v)
    i = lo_v
    j = t - i
    valid = (d0 + t64) < (m + p)
    has_a = valid & (i < na)
    has_b = valid & (j < nb)
    av = tl.load(a + (i0 + i), mask=has_a, other=0)
    bv = tl.load(b + (j0 + j), mask=has_b, other=0)
    take_a = has_a & ((has_b == 0) | (av <= bv))
    tl.store(out + (d0 + t64), tl.where(take_a, av, bv), mask=valid)


def run(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, m: int, p: int) -> None:
    grid = (triton.cdiv(m + p, BLOCK),)
    kernel[grid](a, b, out, m, p, BLOCK=BLOCK, STEPS=STEPS, num_warps=NUM_WARPS)
