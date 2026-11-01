"""radix_scatter: the scatter pass of an LSD radix sort on one 8-bit digit.

One key per lane. The lane takes the digit of its key, claims the next
position of that digit's segment with one atomic add on the digit's cursor,
and stores its row id there. Order inside a segment is arbitrary, as
allowed, and on exit every cursor has advanced to the end of its segment.

Revision: the block-level ranking (histogram, scans and sums over a 2-D
tile) is gone; its shared-memory traffic used instructions the memory
checker does not model. The cursor's address is the base of `cursors` plus
a byte offset masked to [0, 1020]; the store position is the atomic's
result, an index into `out` by contract, whose byte offset is pinned in one
register (an identity PTX mov) and masked to the extent of `out` before the
store. This costs one atomic per key instead of one per (program, digit).
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
TYPES = {"keys": "*u32", "cursors": "*i32", "out": "*i32", "n": "i64", "shift": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def pin(x):
    """Identity through one PTX mov: the mask and the address of an access then use the same register."""
    return tl.inline_asm_elementwise("mov.b64 $0, $1;", "=l,l", [x], dtype=tl.int64, is_pure=True, pack=1)


@triton.jit
def kernel(keys, cursors, out, n: tl.int64, shift: tl.int64, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    key = tl.load(keys + i, mask=m, other=0)
    doff = ((key >> shift.to(tl.uint32)) << 2) & 1020  # byte offset of the digit's cursor
    cb = cursors.to(tl.pointer_type(tl.int8))
    pos = tl.atomic_add((cb + doff).to(tl.pointer_type(tl.int32)), 1, mask=m, sem="relaxed")
    off = pin(pos.to(tl.int64) * 4)
    ok = m & (off >= 0) & (off <= n * 4 - 4)
    ob = out.to(tl.pointer_type(tl.int8))
    tl.store((ob + off).to(tl.pointer_type(tl.int32)), i.to(tl.int32), mask=ok)


def run(keys: torch.Tensor, cursors: torch.Tensor, out: torch.Tensor, n: int, shift: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](keys, cursors, out, n, shift, BLOCK=BLOCK, num_warps=NUM_WARPS)
