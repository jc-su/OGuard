import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(values, first, out, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    row_ids = block_start + tl.arange(0, BLOCK_SIZE)

    row_mask = row_ids < n

    # Load first array for current rows
    first_vals = tl.load(first + row_ids, mask=row_mask, other=0)

    # Compute the source index: row_ids - k
    src_idx = row_ids - k

    # Check if source index is valid: src_idx >= first_vals
    valid = (src_idx >= first_vals)

    # Load values from source indices where valid
    result = tl.load(values + src_idx, mask=valid, other=float('nan'))

    # For invalid indices, store NaN (already loaded as NaN)
    tl.store(out + row_ids, result, mask=row_mask)


TYPES = {
    "values": "*fp64",
    "first": "*i32",
    "out": "*fp64",
    "n": "i64",
    "k": "i64",
}

CONSTANTS = {}

NUM_WARPS = 4


def run(values, first, out, n, k):
    grid = (triton.cdiv(n, 256),)
    kernel[grid](values, first, out, n, k, BLOCK_SIZE=256, num_warps=NUM_WARPS)
