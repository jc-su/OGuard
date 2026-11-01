from numba import cuda, types
import numpy as np

TYPES = (types.int64[:], types.uint32[:], types.int8[:], types.int64, types.int64)

def kernel(keys, bits, match, n, words):
    """Probe a Bloom filter: match[i] = 1 if both bits are set, 0 otherwise."""
    idx = cuda.grid(1)
    if idx < n:
        key = keys[idx]

        # Hash functions
        h1 = ((key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
        h2 = ((key * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF) >> 32

        # Check bits
        bit_count = 32 * words
        bit_pos1 = h1 % bit_count
        bit_pos2 = h2 % bit_count

        word_idx1 = bit_pos1 // 32
        bit_idx1 = bit_pos1 % 32
        is_set1 = (bits[word_idx1] >> bit_idx1) & 1

        word_idx2 = bit_pos2 // 32
        bit_idx2 = bit_pos2 % 32
        is_set2 = (bits[word_idx2] >> bit_idx2) & 1

        match[idx] = 1 if (is_set1 and is_set2) else 0

def run(keys, bits, n, words):
    """Launch the bloom_probe kernel on device arrays."""
    keys_dev = cuda.to_device(keys)
    bits_dev = cuda.to_device(bits)
    match_dev = cuda.device_array(n, dtype=np.int8)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](keys_dev, bits_dev, match_dev, n, words)

    return match_dev.copy_to_host()
