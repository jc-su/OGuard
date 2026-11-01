"""radix_scatter: the scatter pass of an LSD radix sort on one 8-bit digit.

Each program ranks its BLOCK keys locally so that the 256 cursors see one
atomic add per (program, non-empty digit) instead of one per key:

1. histogram of the digits of the program's keys, and its exclusive scan
   (the start of each digit's segment inside the program);
2. one masked atomic add per digit reserves the program's segments in ``out``
   and returns their global starts;
3. the keys are sorted by (digit, local index) packed into one int32, which
   groups equal digits and keeps their input order; the sorted position of a
   key minus its digit's local segment start is its rank within the digit, so
   its output position is sorted position + (global start - local start) of
   its digit, gathered per key from the 256-entry table.
"""

import torch
import triton
import triton.language as tl

BLOCK = 2048
LOCAL_BITS = 11  # BLOCK == 1 << LOCAL_BITS
NUM_WARPS = 8
TYPES = {"keys": "*u32", "cursors": "*i32", "out": "*i32", "n": "i64", "shift": "i64"}
CONSTANTS = {"BLOCK": BLOCK, "LOCAL_BITS": LOCAL_BITS}


@triton.jit
def kernel(keys, cursors, out, n: tl.int64, shift: tl.int64, BLOCK: tl.constexpr, LOCAL_BITS: tl.constexpr):
    tl.static_assert(BLOCK == 1 << LOCAL_BITS)
    start = tl.program_id(0).to(tl.int64) * BLOCK
    local = tl.arange(0, BLOCK)
    i = start + local
    m = i < n
    key = tl.load(keys + i, mask=m, other=0)
    digit = ((key >> shift.to(tl.uint32)) & 255).to(tl.int32)
    hist = tl.histogram(digit, 256, mask=m)
    local_start = tl.cumsum(hist, axis=0) - hist
    global_start = tl.atomic_add(cursors + tl.arange(0, 256), hist, mask=hist > 0, sem="relaxed")
    delta = global_start - local_start
    # rows past n get the sentinel, sort to the end and are never stored;
    # their digit field is clamped so the gather stays inside the table
    packed = tl.where(m, (digit << LOCAL_BITS) | local, 2147483647)
    s = tl.sort(packed)
    s_digit = tl.minimum(s >> LOCAL_BITS, 255)
    s_local = s & (BLOCK - 1)
    pos = local + tl.gather(delta, s_digit, axis=0)
    tl.store(out + pos, (start + s_local).to(tl.int32), mask=s != 2147483647)


def run(keys: torch.Tensor, cursors: torch.Tensor, out: torch.Tensor, n: int, shift: int) -> None:
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](keys, cursors, out, n, shift, BLOCK=BLOCK, LOCAL_BITS=LOCAL_BITS, num_warps=NUM_WARPS)
