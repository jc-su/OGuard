import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(a_offsets, a_chars, b_offsets, b_chars, match, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    row_ids = block_start + tl.arange(0, BLOCK_SIZE)

    row_mask = row_ids < n

    # Load offsets for current rows
    a_start = tl.load(a_offsets + row_ids, mask=row_mask, other=0)
    a_end = tl.load(a_offsets + row_ids + 1, mask=row_mask, other=0)
    b_start = tl.load(b_offsets + row_ids, mask=row_mask, other=0)
    b_end = tl.load(b_offsets + row_ids + 1, mask=row_mask, other=0)

    # Check if lengths match
    a_len = a_end - a_start
    b_len = b_end - b_start

    len_match = a_len == b_len

    # Initialize result: 1 if lengths match, 0 otherwise
    match_result = tl.where(len_match, tl.cast(1, tl.int8), tl.cast(0, tl.int8))

    # Compare byte by byte
    # Iterate through possible byte positions (up to max string length)
    for char_idx in range(0, 4096):  # Reasonable max string length
        # Check if this byte exists in both strings
        char_in_range = char_idx < a_len

        # Only load and compare if lengths matched so far
        compare_mask = len_match & char_in_range & row_mask

        # Load characters
        a_char = tl.load(a_chars + a_start + char_idx, mask=compare_mask, other=0)
        b_char = tl.load(b_chars + b_start + char_idx, mask=compare_mask, other=0)

        # Check if characters match
        chars_match = (a_char == b_char)

        # If any character doesn't match, mark as not equal
        match_result = tl.where(
            compare_mask,
            tl.where(chars_match, match_result, tl.cast(0, tl.int8)),
            match_result
        )

    # Store results
    tl.store(match + row_ids, match_result, mask=row_mask)


TYPES = {
    "a_offsets": "*i32",
    "a_chars": "*u8",
    "b_offsets": "*i32",
    "b_chars": "*u8",
    "match": "*i8",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(a_offsets, a_chars, b_offsets, b_chars, match, n):
    grid = (triton.cdiv(n, CONSTANTS["BLOCK_SIZE"]),)
    kernel[grid](a_offsets, a_chars, b_offsets, b_chars, match, n, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
