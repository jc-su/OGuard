import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/bloom_probe')
from numba import cuda
from numba_module import kernel, TYPES, run

def hash_bloom(key):
    """Compute hash values for Bloom filter."""
    h1 = ((key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
    h2 = ((key * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF) >> 32
    return h1, h2

def test_bloom_probe():
    test_configs = [(0, 4), (1, 4), (10, 8), (1000, 16)]

    for n, words in test_configs:
        print(f"Testing bloom_probe with n={n}, words={words}")

        # Build filter with some keys
        build_keys = np.arange(n // 2, dtype=np.int64)
        bits = np.zeros(words, dtype=np.uint32)

        for i in range(len(build_keys)):
            h1, h2 = hash_bloom(build_keys[i])
            bit_count = 32 * words
            bit_pos1 = h1 % bit_count
            bit_pos2 = h2 % bit_count

            word_idx1 = bit_pos1 // 32
            bit_idx1 = bit_pos1 % 32
            bits[word_idx1] |= (1 << bit_idx1)

            if bit_pos2 != bit_pos1:
                word_idx2 = bit_pos2 // 32
                bit_idx2 = bit_pos2 % 32
                bits[word_idx2] |= (1 << bit_idx2)

        # Probe with different keys
        probe_keys = np.arange(n, dtype=np.int64)

        # Expected result: 1 if key was in build set, 0 otherwise
        # (Assuming no false positives for this test)
        expected = np.zeros(n, dtype=np.int8)
        for i in range(n):
            if i < len(build_keys):
                expected[i] = 1

        # Run kernel
        result = run(probe_keys, bits, n, words)

        # Compare (note: Bloom filter may have false positives)
        matches = 0
        for i in range(n):
            if i < len(build_keys) and result[i] == 1:
                matches += 1

        # All built keys should be found (no false negatives)
        no_false_negatives = True
        for i in range(min(n, len(build_keys))):
            if result[i] != 1:
                no_false_negatives = False

        if no_false_negatives:
            print(f"  PASS (no false negatives)")
        else:
            print(f"  FAIL")
            if n <= 10:
                print(f"    result: {result}")
                print(f"    expected: {expected}")

if __name__ == "__main__":
    test_bloom_probe()
