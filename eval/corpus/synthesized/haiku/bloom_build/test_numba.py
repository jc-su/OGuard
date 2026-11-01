import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/bloom_build')
from numba import cuda
from numba_module import kernel, TYPES, run

def hash_bloom(key):
    """Compute hash values for Bloom filter."""
    h1 = ((key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
    h2 = ((key * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF) >> 32
    return h1, h2

def test_bloom_build():
    test_configs = [(0, 4), (1, 4), (10, 8), (1000, 16)]

    for n, words in test_configs:
        print(f"Testing bloom_build with n={n}, words={words}")

        # Create keys
        keys = np.arange(n, dtype=np.int64)

        # Expected: bits set for each key
        expected = np.zeros(words, dtype=np.uint32)
        for i in range(n):
            h1, h2 = hash_bloom(keys[i])
            bit_count = 32 * words
            bit_pos1 = h1 % bit_count
            bit_pos2 = h2 % bit_count

            word_idx1 = bit_pos1 // 32
            bit_idx1 = bit_pos1 % 32
            expected[word_idx1] |= (1 << bit_idx1)

            if bit_pos2 != bit_pos1:
                word_idx2 = bit_pos2 // 32
                bit_idx2 = bit_pos2 % 32
                expected[word_idx2] |= (1 << bit_idx2)

        # Run kernel
        result = run(keys, n, words)

        # Compare
        if np.array_equal(result, expected):
            print(f"  PASS")
        else:
            print(f"  FAIL")
            print(f"    result: {result}")
            print(f"    expected: {expected}")

if __name__ == "__main__":
    test_bloom_build()
