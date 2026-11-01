"""Triton kernel for filter_select.

Writes the row ids i where quantity[i] < threshold into sel, in any
order, and the number of matches into count.

Precondition: the caller zeroes count before each launch.
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
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    q = tl.load(quantity + offs, mask=mask, other=0.0)
    hit = mask & (q < threshold)
    count_block = count + tl.zeros((BLOCK,), dtype=tl.int32)
    pos = tl.atomic_add(count_block, 1, mask=hit)
    tl.store(sel + pos, offs, mask=hit)


def run(quantity, sel, count, threshold, n):
    """Launch kernel on PyTorch CUDA tensors. count must already be zeroed."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](
        quantity, sel, count, threshold, n, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS
    )
