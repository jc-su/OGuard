"""merge_sorted: merge path merge of two sorted int64 columns.

Every lane owns one output position d and finds the partition of that
diagonal, the number i of a-elements among the first d outputs, with a
binary search over [0, d] on the global arrays; then it emits a[i] or
b[d - i], a first on ties. The search steps are unrolled at compile time
(STEPS halvings cover any d below 2^41), and a probe past the end of an
input counts as an exhausted input, so no min/max clamps the bounds. Lanes
with neighbouring diagonals probe neighbouring positions, so the gathers of
a warp fall into few cache lines.

Revision: the k-ary search of the program's window and the per-lane search
inside it are replaced by this one search per lane, whose loop is static and
whose probes are gathers with a mask on the exact byte offset that forms
the address (the offset is pinned in one register by an identity PTX mov).
The final loads use the same form. Under the task's contract the extra
mask terms never exclude a lane.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
STEPS = 42  # halvings of a search interval of at most 2^41 positions
NUM_WARPS = 4
TYPES = {"a": "*i64", "b": "*i64", "out": "*i64", "m": "i64", "p": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "STEPS": STEPS}


@triton.jit
def pin(x):
    """Identity through one PTX mov: the mask and the address of a gather then use the same register."""
    return tl.inline_asm_elementwise("mov.b64 $0, $1;", "=l,l", [x], dtype=tl.int64, is_pure=True, pack=1)


@triton.jit
def kernel(a, b, out, m: tl.int64, p: tl.int64, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    d = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = d < m + p
    ab = a.to(tl.pointer_type(tl.int8))
    bb = b.to(tl.pointer_type(tl.int8))
    last_a = m * 8 - 8  # byte offset of the last element of a
    last_b = p * 8 - 8

    # smallest i in [0, d] with a[i] > b[d - 1 - i]
    lo = tl.zeros([BLOCK], tl.int64)
    hi = d
    for _ in tl.static_range(STEPS):
        active = lo < hi
        mid = lo + ((hi - lo) >> 1)
        jb = d - 1 - mid
        offa = pin(mid * 8)
        offb = pin(jb * 8)
        oka = active & (offa >= 0) & (offa <= last_a)
        okb = active & (offb >= 0) & (offb <= last_b)
        av = tl.load((ab + offa).to(tl.pointer_type(tl.int64)), mask=oka, other=0)
        bv = tl.load((bb + offb).to(tl.pointer_type(tl.int64)), mask=okb, other=0)
        take_a = oka & ((okb == 0) | (av <= bv))  # an exhausted b counts as +inf, an exhausted a too
        lo = tl.where(active & take_a, mid + 1, lo)
        hi = tl.where(active & (take_a == 0), mid, hi)

    i = lo
    j = d - lo
    offi = pin(i * 8)
    offj = pin(j * 8)
    has_a = valid & (offi >= 0) & (offi <= last_a)
    has_b = valid & (offj >= 0) & (offj <= last_b)
    av = tl.load((ab + offi).to(tl.pointer_type(tl.int64)), mask=has_a, other=0)
    bv = tl.load((bb + offj).to(tl.pointer_type(tl.int64)), mask=has_b, other=0)
    take_a = has_a & ((has_b == 0) | (av <= bv))
    tl.store(out + d, tl.where(take_a, av, bv), mask=valid)


def run(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, m: int, p: int) -> None:
    grid = (triton.cdiv(m + p, BLOCK),)
    kernel[grid](a, b, out, m, p, BLOCK=BLOCK, STEPS=STEPS, num_warps=NUM_WARPS)
