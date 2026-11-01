import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(a, valid, b, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n

    # Load validity bitmap
    byte_offsets = offsets // 8
    bit_offsets = offsets % 8

    # Bounds check for valid array
    byte_mask = byte_offsets < (n + 7) // 8
    valid_data = tl.load(valid + byte_offsets, mask=byte_mask, other=0)

    # Extract bit for each position
    bit_set = ((valid_data >> bit_offsets.to(tl.uint32)) & 1).to(tl.int1)

    # Load a and b
    a_vals = tl.load(a + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b + offsets, mask=mask, other=0.0)

    # Select based on validity
    result = tl.where(bit_set, a_vals, b_vals)

    tl.store(out + offsets, result, mask=mask)


TYPES = {
    "a": "*fp64",
    "valid": "*u8",
    "b": "*fp64",
    "out": "*fp64",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(a, valid, b, out, n):
    grid = (triton.cdiv(n, CONSTANTS["BLOCK_SIZE"]),)
    kernel[grid](a, valid, b, out, n, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
