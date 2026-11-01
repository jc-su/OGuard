import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(values, bounds, out, n, b, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n

    # Load values
    vals = tl.load(values + offsets, mask=mask, other=0.0)

    # For each value, count how many bounds are <= value
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # Iterate through all bounds
    for bound_idx in range(b):
        # Load single bound value
        bound_val = tl.load(bounds + bound_idx)

        # Check if bound_val <= each value
        is_leq = bound_val <= vals

        # Accumulate count
        count = count + is_leq.to(tl.int32)

    tl.store(out + offsets, count, mask=mask)


TYPES = {
    "values": "*fp64",
    "bounds": "*fp64",
    "out": "*i32",
    "n": "i64",
    "b": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(values, bounds, out, n, b):
    grid = (triton.cdiv(n, 256),)
    kernel[grid](values, bounds, out, n, b, BLOCK_SIZE=256, num_warps=NUM_WARPS)
