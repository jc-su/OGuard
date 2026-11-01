"""Triton kernel for filter_select.

Writes the row ids i where quantity[i] < threshold into sel, in any
order, and the number of matches into count.

Repair: two issues kept the static checker from proving every access
stays inside its buffer.

1. offs was computed as pid * BLOCK + arange in 32-bit arithmetic.
   Since the checker must prove safety for any grid, pid is not
   bounded by the grid this file happens to launch, and pid * BLOCK
   can overflow a 32-bit index and wrap into a value that still
   passes `offs < n` while addressing outside quantity/sel. offs is
   now built in 64-bit arithmetic throughout, so no such wraparound
   is possible.
2. The store into sel was addressed by pos, the value returned by an
   atomic add on count. The task's contract does not promise count is
   zero on entry (unlike, for example, hash_probe's out_count), so
   the checker must assume count can hold anything on entry, which
   makes pos unbounded. The store is now masked so it only fires when
   0 <= pos < n, which keeps every write inside sel regardless of
   count's starting value. For the real use (count zeroed by the
   caller, as tested here), pos is always in [0, n) already, so this
   mask changes nothing about the result.
"""

import triton
import triton.language as tl

TYPES = {
    "quantity": "*fp64",
    "sel": "*i32",
    "count": "*i32",
    "threshold": "fp64",
    "n": "i64",
}

CONSTANTS = {"BLOCK": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(quantity, sel, count, threshold, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    q = tl.load(quantity + offs, mask=mask, other=0.0)
    hit = mask & (q < threshold)

    count_block = count + tl.zeros((BLOCK,), dtype=tl.int32)
    pos = tl.atomic_add(count_block, 1, mask=hit).to(tl.int64)
    store_mask = hit & (pos >= 0) & (pos < n)
    tl.store(sel + pos, offs.to(tl.int32), mask=store_mask)


def run(quantity, sel, count, threshold, n):
    """Launch kernel on PyTorch CUDA tensors. count must already be zeroed."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](
        quantity, sel, count, threshold, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS
    )
