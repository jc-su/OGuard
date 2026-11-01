import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def kernel(keys, bits, n, words, BLOCK_SIZE: tl.constexpr):
    # Single thread block processes all keys
    pid = tl.program_id(0)

    # Each thread processes a subset of keys
    for key_id in range(pid, n, BLOCK_SIZE):
        # Load one key
        key_val = tl.load(keys + key_id)

        # Compute hashes
        h1_u64 = ((key_val.to(tl.uint64) * tl.cast(0x9E3779B97F4A7C15, tl.uint64)) >> 32)
        h2_u64 = ((key_val.to(tl.uint64) * tl.cast(0xC2B2AE3D27D4EB4F, tl.uint64)) >> 32)

        total_bits = tl.cast(32 * words, tl.uint64)
        bit_pos_1 = h1_u64 % total_bits
        bit_pos_2 = h2_u64 % total_bits

        # Process both bit positions
        for hash_idx in range(2):
            if hash_idx == 0:
                bit_pos = bit_pos_1
            else:
                bit_pos = bit_pos_2

            word_idx = bit_pos // 32
            bit_idx = (bit_pos % 32).to(tl.uint32)
            bit_mask = tl.cast(1, tl.uint32) << bit_idx

            # Atomic OR using CAS loop (no break supported)
            # Retry a few times - usually succeeds on first try
            for _ in range(50):
                old_val = tl.load(bits + word_idx)
                new_val = old_val | bit_mask
                # Try CAS (if it succeeds, new_val is stored, else loop retries)
                tl.atomic_cas(bits + word_idx, old_val, new_val)


TYPES = {
    "keys": "*i64",
    "bits": "*u32",
    "n": "i64",
    "words": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(keys, bits, n, words):
    grid = (CONSTANTS["BLOCK_SIZE"],)  # One block per BLOCK_SIZE threads
    kernel[grid](keys, bits, n, words, BLOCK_SIZE=CONSTANTS["BLOCK_SIZE"], num_warps=NUM_WARPS)
