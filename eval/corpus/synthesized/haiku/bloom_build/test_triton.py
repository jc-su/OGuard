import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/bloom_build/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_bloom_build(n, words):
    # Create test data
    keys = np.random.randint(0, 2**63, n, dtype=np.int64)

    # Compute CPU reference
    bits_ref = np.zeros(words, dtype=np.uint32)

    for key in keys:
        # Compute hashes (unsigned 64-bit arithmetic with wraparound)
        # Use Python's built-in integer type which doesn't overflow
        key_int = int(key) & 0xFFFFFFFFFFFFFFFF  # ensure 64-bit unsigned
        h1_int = ((key_int * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
        h2_int = ((key_int * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF) >> 32
        h1 = np.uint64(h1_int)
        h2 = np.uint64(h2_int)

        total_bits = 32 * words
        bit_pos_1 = h1 % total_bits
        bit_pos_2 = h2 % total_bits

        # Set bits
        for bit_pos in [bit_pos_1, bit_pos_2]:
            word_idx = bit_pos // 32
            bit_idx = bit_pos % 32
            bits_ref[word_idx] |= (1 << bit_idx)

    # GPU computation
    device = torch.device('cuda:0')
    keys_gpu = torch.from_numpy(keys).to(device)
    bits_gpu = torch.zeros(words, dtype=torch.uint32, device=device)

    run(keys_gpu, bits_gpu, n, words)

    bits_result = bits_gpu.cpu().numpy()

    # Check results
    if not np.array_equal(bits_result, bits_ref):
        print(f"FAILED for n={n}, words={words}")
        diff_idx = np.where(bits_result != bits_ref)[0]
        if len(diff_idx) > 0:
            idx = diff_idx[0]
            print(f"First mismatch at word {idx}: got {bits_result[idx]}, expected {bits_ref[idx]}")
        return False
    print(f"PASSED for n={n}, words={words}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n in [0, 1, 1000, 100000]:
        words = 256  # 2^8
        all_pass = all_pass and test_bloom_build(n, words)

    if all_pass:
        print("\nAll bloom_build tests passed!")
    else:
        print("\nSome bloom_build tests failed!")
        sys.exit(1)
