import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util

# Load bloom_build to reuse for test setup
spec_build = importlib.util.spec_from_file_location(
    "bloom_build",
    "/opgate/eval/corpus/synthesized/haiku/bloom_build/triton.py"
)
bloom_build_module = importlib.util.module_from_spec(spec_build)
sys.modules["bloom_build"] = bloom_build_module
spec_build.loader.exec_module(bloom_build_module)
run_build = bloom_build_module.run

# Load bloom_probe
spec = importlib.util.spec_from_file_location(
    "kernel_module",
    "/opgate/eval/corpus/synthesized/haiku/bloom_probe/triton.py"
)
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run_probe = kernel_module.run

def test_bloom_probe(n_keys, n_probe, words):
    # Create training keys
    keys_train = np.random.randint(0, 2**63, n_keys, dtype=np.int64)

    # Build Bloom filter
    device = torch.device('cuda:0')
    keys_train_gpu = torch.from_numpy(keys_train).to(device)
    bits_gpu = torch.zeros(words, dtype=torch.uint32, device=device)

    run_build(keys_train_gpu, bits_gpu, n_keys, words)

    # Create probe keys - mix of keys in the training set and random keys
    keys_probe = []
    for i in range(n_probe):
        if i < n_keys and np.random.rand() < 0.5:
            # Use a training key
            keys_probe.append(keys_train[i])
        else:
            # Use a random key
            keys_probe.append(np.random.randint(0, 2**63, dtype=np.int64))

    keys_probe = np.array(keys_probe, dtype=np.int64)

    # Compute CPU reference
    match_ref = np.zeros(n_probe, dtype=np.int8)
    for i in range(n_probe):
        key = keys_probe[i]
        key_u64 = np.uint64(key) & 0xFFFFFFFFFFFFFFFF
        h1_int = ((key_u64 * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> 32
        h2_int = ((key_u64 * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF) >> 32
        h1 = np.uint64(h1_int)
        h2 = np.uint64(h2_int)

        total_bits = 32 * words
        bit_pos_1 = h1 % total_bits
        bit_pos_2 = h2 % total_bits

        word_idx_1 = bit_pos_1 // 32
        word_idx_2 = bit_pos_2 // 32
        bit_idx_1 = bit_pos_1 % 32
        bit_idx_2 = bit_pos_2 % 32

        bits_np = bits_gpu.cpu().numpy()
        bit_1 = (bits_np[word_idx_1] >> bit_idx_1) & 1
        bit_2 = (bits_np[word_idx_2] >> bit_idx_2) & 1

        match_ref[i] = 1 if (bit_1 & bit_2) else 0

    # GPU computation
    keys_probe_gpu = torch.from_numpy(keys_probe).to(device)
    match_gpu = torch.zeros(n_probe, dtype=torch.int8, device=device)

    run_probe(keys_probe_gpu, bits_gpu, match_gpu, n_probe, words)

    match_result = match_gpu.cpu().numpy()

    # Check results
    if not np.array_equal(match_result, match_ref):
        print(f"FAILED for n_probe={n_probe}, words={words}")
        diff_idx = np.where(match_result != match_ref)[0]
        if len(diff_idx) > 0:
            idx = diff_idx[0]
            print(f"First mismatch at index {idx}: got {match_result[idx]}, expected {match_ref[idx]}")
        return False
    print(f"PASSED for n_probe={n_probe}, words={words}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n_probe in [0, 1, 1000, 100000]:
        n_keys = max(1, n_probe // 2)
        words = 256
        all_pass = all_pass and test_bloom_probe(n_keys, n_probe, words)

    if all_pass:
        print("\nAll bloom_probe tests passed!")
    else:
        print("\nSome bloom_probe tests failed!")
        sys.exit(1)
