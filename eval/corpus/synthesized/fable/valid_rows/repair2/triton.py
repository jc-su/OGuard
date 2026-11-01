"""valid_rows: compact the ids of the valid rows of an Arrow validity bitmap.

Each program handles BLOCK bitmap bytes (8 * BLOCK rows). Every lane loads
one byte and, for each of its eight rows that exists and whose bit is set,
claims the next slot of ``rows`` with one atomic add on ``count`` and stores
the row id there. Output order is the order of the atomics.

Revision: the block-wide prefix sum and the single atomic per program are
replaced by one atomic per valid row, so no store position passes through
shared memory; the position's byte offset is pinned in one register (an
identity PTX mov) and masked to the extent of ``rows`` before the store.
The byte index is affine in the program and lane ids and is masked to the
bitmap size (n + 7) >> 3.
"""

import torch
import triton
import triton.language as tl

BLOCK = 256  # bitmap bytes per program
NUM_WARPS = 4
TYPES = {"valid": "*u8", "rows": "*i32", "count": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def pin(x):
    """Identity through one PTX mov: the mask and the address of an access then use the same register."""
    return tl.inline_asm_elementwise("mov.b64 $0, $1;", "=l,l", [x], dtype=tl.int64, is_pure=True, pack=1)


@triton.jit
def kernel(valid, rows, count, n: tl.int64, BLOCK: tl.constexpr):
    j = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # bitmap byte
    nbytes = (n + 7) >> 3
    okb = j < nbytes
    bits = tl.load(valid + j, mask=okb, other=0)
    counter = count + tl.zeros([BLOCK], tl.int32)
    rb = rows.to(tl.pointer_type(tl.int8))
    last = n * 4 - 4  # byte offset of the last slot of rows
    for bit in tl.static_range(8):
        row = j * 8 + bit
        v = okb & (row < n) & (((bits >> bit) & 1) != 0)
        pos = tl.atomic_add(counter, 1, mask=v, sem="relaxed")
        off = pin(pos.to(tl.int64) * 4)
        ok = v & (off >= 0) & (off <= last)
        tl.store((rb + off).to(tl.pointer_type(tl.int32)), row.to(tl.int32), mask=ok)


def run(valid: torch.Tensor, rows: torch.Tensor, count: torch.Tensor, n: int) -> None:
    nbytes = (n + 7) // 8
    grid = (triton.cdiv(nbytes, BLOCK),)
    kernel[grid](valid, rows, count, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
