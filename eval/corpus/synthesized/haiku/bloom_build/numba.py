from numba import cuda, types
import numpy as np

TYPES = (types.int64[:], types.uint32[:], types.int64, types.int64)

def kernel(keys, bits, n, words):
    """Build a Bloom filter by setting bits for each key."""
    idx = cuda.grid(1)
    if idx < n:
        key = keys[idx]

        # Hash functions
        h1 = ((key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
        h2 = ((key * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF) >> 32

        # Set bits
        bit_count = 32 * words
        bit_pos1 = h1 % bit_count
        bit_pos2 = h2 % bit_count

        word_idx1 = bit_pos1 // 32
        bit_idx1 = bit_pos1 % 32
        cuda.atomic.add(bits, word_idx1, 1 << bit_idx1)

        if bit_pos2 != bit_pos1:
            word_idx2 = bit_pos2 // 32
            bit_idx2 = bit_pos2 % 32
            cuda.atomic.add(bits, word_idx2, 1 << bit_idx2)

def run(keys, n, words):
    """Launch the bloom_build kernel on device arrays."""
    keys_dev = cuda.to_device(keys)
    bits_dev = cuda.device_array(words, dtype=np.uint32)
    # Initialize to zero
    cuda.to_device(np.zeros(words, dtype=np.uint32), bits_dev)

    # Launch kernel
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    cuda.jit(kernel)[blocks, threads_per_block](keys_dev, bits_dev, n, words)

    return bits_dev.copy_to_host()
