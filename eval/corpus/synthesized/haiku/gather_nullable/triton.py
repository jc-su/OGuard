import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(col, idx, out, m, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < m

    # Load indices
    indices = tl.load(idx + offsets, mask=mask, other=-1)

    # Create mask for valid indices
    valid_idx_mask = indices >= 0

    # Load values from col, using 0 as placeholder for invalid indices
    col_vals = tl.load(col + indices, mask=valid_idx_mask, other=0.0)

    # Convert to NaN where indices were -1
    # NaN in float64 is represented as a specific bit pattern
    nan_val = tl.where(valid_idx_mask, col_vals, float('nan'))

    tl.store(out + offsets, nan_val, mask=mask)


TYPES = {
    "col": "*fp64",
    "idx": "*i32",
    "out": "*fp64",
    "m": "i64",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(col, idx, out, m, n):
    grid = (triton.cdiv(m, CONSTANTS["BLOCK_SIZE"]),)
    kernel[grid](col, idx, out, m, n, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
