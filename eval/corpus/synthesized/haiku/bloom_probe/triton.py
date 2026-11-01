import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(keys, bits, match, n, words, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    key_ids = block_start + tl.arange(0, BLOCK_SIZE)

    key_mask = key_ids < n

    # Load keys
    key_vals = tl.load(keys + key_ids, mask=key_mask, other=0)

    # Compute hashes
    h1_u64 = ((key_vals.to(tl.uint64) * tl.cast(0x9E3779B97F4A7C15, tl.uint64)) >> 32)
    h2_u64 = ((key_vals.to(tl.uint64) * tl.cast(0xC2B2AE3D27D4EB4F, tl.uint64)) >> 32)

    total_bits = tl.cast(32 * words, tl.uint64)
    bit_pos_1 = h1_u64 % total_bits
    bit_pos_2 = h2_u64 % total_bits

    # Extract word indices and bit indices
    word_idx_1 = bit_pos_1 // 32
    word_idx_2 = bit_pos_2 // 32
    bit_idx_1 = (bit_pos_1 % 32).to(tl.uint32)
    bit_idx_2 = (bit_pos_2 % 32).to(tl.uint32)

    # Load words from bits array
    bits_word_1 = tl.load(bits + word_idx_1, mask=key_mask, other=0)
    bits_word_2 = tl.load(bits + word_idx_2, mask=key_mask, other=0)

    # Extract bits
    bit_1 = (bits_word_1 >> bit_idx_1) & 1
    bit_2 = (bits_word_2 >> bit_idx_2) & 1

    # Match if both bits are set
    result = tl.where((bit_1 & bit_2).to(tl.int1), tl.cast(1, tl.int8), tl.cast(0, tl.int8))

    tl.store(match + key_ids, result, mask=key_mask)


TYPES = {
    "keys": "*i64",
    "bits": "*u32",
    "match": "*i8",
    "n": "i64",
    "words": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(keys, bits, match, n, words):
    grid = (triton.cdiv(n, 256),)
    kernel[grid](keys, bits, match, n, words, BLOCK_SIZE=256, num_warps=NUM_WARPS)
