import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/string_equal/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_string_equal(n):
    # Create test data: two string columns
    # Each string is a random sequence of ASCII characters

    # Build first column
    a_chars_list = []
    a_offsets = [0]
    for i in range(n):
        # Random string of 0-100 characters
        str_len = np.random.randint(0, 100)
        chars = np.random.randint(32, 127, str_len, dtype=np.uint8)
        a_chars_list.append(chars)
        a_offsets.append(a_offsets[-1] + str_len)

    a_chars = np.concatenate(a_chars_list) if a_chars_list else np.array([], dtype=np.uint8)
    a_offsets = np.array(a_offsets, dtype=np.int32)

    # Build second column - some same, some different
    b_chars_list = []
    b_offsets = [0]
    for i in range(n):
        if np.random.rand() < 0.5 and i < len(a_chars_list):
            # Copy from first column
            chars = a_chars_list[i].copy()
        else:
            # Different string
            str_len = np.random.randint(0, 100)
            chars = np.random.randint(32, 127, str_len, dtype=np.uint8)
        b_chars_list.append(chars)
        b_offsets.append(b_offsets[-1] + len(chars))

    b_chars = np.concatenate(b_chars_list) if b_chars_list else np.array([], dtype=np.uint8)
    b_offsets = np.array(b_offsets, dtype=np.int32)

    # Compute CPU reference
    match_ref = np.zeros(n, dtype=np.int8)
    for i in range(n):
        a_str = a_chars[a_offsets[i]:a_offsets[i+1]]
        b_str = b_chars[b_offsets[i]:b_offsets[i+1]]
        if len(a_str) == len(b_str) and np.array_equal(a_str, b_str):
            match_ref[i] = 1
        else:
            match_ref[i] = 0

    # GPU computation
    device = torch.device('cuda:0')
    a_offsets_gpu = torch.from_numpy(a_offsets).to(device)
    a_chars_gpu = torch.from_numpy(a_chars).to(device)
    b_offsets_gpu = torch.from_numpy(b_offsets).to(device)
    b_chars_gpu = torch.from_numpy(b_chars).to(device)
    match_gpu = torch.zeros(n, dtype=torch.int8, device=device)

    run(a_offsets_gpu, a_chars_gpu, b_offsets_gpu, b_chars_gpu, match_gpu, n)

    match_result = match_gpu.cpu().numpy()

    # Check results
    if not np.array_equal(match_result, match_ref):
        print(f"FAILED for n={n}")
        diff_idx = np.where(match_result != match_ref)[0]
        if len(diff_idx) > 0:
            idx = diff_idx[0]
            print(f"First mismatch at index {idx}: got {match_result[idx]}, expected {match_ref[idx]}")
            a_str = a_chars[a_offsets[idx]:a_offsets[idx+1]]
            b_str = b_chars[b_offsets[idx]:b_offsets[idx+1]]
            print(f"  String A: {a_str}")
            print(f"  String B: {b_str}")
        return False
    print(f"PASSED for n={n}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n in [0, 1, 1000, 1000003]:
        all_pass = all_pass and test_string_equal(n)

    if all_pass:
        print("\nAll string_equal tests passed!")
    else:
        print("\nSome string_equal tests failed!")
        sys.exit(1)
