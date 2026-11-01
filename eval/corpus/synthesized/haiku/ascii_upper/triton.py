import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(offsets, chars, out, n, BLOCK_SIZE: tl.constexpr):
    # Get the total length of the character array from the offsets
    # Process character by character

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    char_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the last offset to get total chars
    # We need to process chars in blocks by their indices
    # Load characters and convert lowercase to uppercase

    # First, determine total length by loading the last offset
    # For now, let's load from a large enough offset
    total_chars = tl.load(offsets + n)

    char_mask = char_offsets < total_chars

    # Load characters
    char_vals = tl.load(chars + char_offsets, mask=char_mask, other=0)

    # Convert lowercase ASCII (a-z = 97-122) to uppercase (A-Z = 65-90)
    # char_vals >= 97 && char_vals <= 122 -> char_vals - 32
    is_lowercase = (char_vals >= 97) & (char_vals <= 122)
    result = tl.where(is_lowercase, char_vals - 32, char_vals)

    tl.store(out + char_offsets, result, mask=char_mask)


TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "out": "*u8",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(offsets, chars, out, n):
    # Get total number of characters
    offsets_np = offsets.cpu().numpy() if isinstance(offsets, torch.Tensor) else offsets
    total_chars = int(offsets_np[-1])

    grid = (triton.cdiv(total_chars, CONSTANTS["BLOCK_SIZE"]),)
    kernel[grid](offsets, chars, out, n, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
