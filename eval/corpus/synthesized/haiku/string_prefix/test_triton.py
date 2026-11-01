#!/usr/bin/env python
import sys
import os

# Remove the script directory from path so we can import real triton
script_dir = os.path.dirname(os.path.abspath(__file__))
original_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir and p != ""]

# Now safely import triton
import triton as _real_triton
import triton.language as _real_tl
import torch
import numpy as np

# Restore path
sys.path = original_path

import importlib.util

# Load our kernel module by path
kernel_path = os.path.join(script_dir, "triton.py")

spec = importlib.util.spec_from_file_location(
    "_string_prefix_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def test_string_prefix(n: int, k: int) -> bool:
    """Test string_prefix kernel."""
    device = torch.device("cuda:0")

    # Create simple test data
    # For each string i, use a pattern of bytes: [i % 256, (i+1) % 256, ...]
    chars_list = []
    offsets = [0]

    for i in range(n):
        string_len = (i % 10) + 1  # Vary string length from 1 to 10
        for j in range(string_len):
            chars_list.append((i + j) % 256)
        offsets.append(len(chars_list))

    # Create tensors
    offsets_np = np.array(offsets, dtype=np.int32)
    chars_np = np.array(chars_list, dtype=np.uint8) if chars_list else np.array([], dtype=np.uint8)

    offsets_t = torch.from_numpy(offsets_np).to(device)
    chars_t = torch.from_numpy(chars_np).to(device) if len(chars_list) > 0 else torch.zeros(0, dtype=torch.uint8, device=device)
    out_t = torch.zeros(n * k, dtype=torch.uint8, device=device)

    # Run kernel
    run(offsets_t, chars_t, out_t, n, k)

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    out_np = out_t.cpu().numpy()

    success = True
    for i in range(min(n, 100)):  # Check first 100 strings
        start = offsets[i]
        end = offsets[i + 1]
        string_len = end - start

        expected = []
        for j in range(k):
            if j < string_len:
                expected.append(chars_list[start + j])
            else:
                expected.append(0)

        actual = out_np[i * k : (i + 1) * k]
        expected_arr = np.array(expected, dtype=np.uint8)

        if not np.array_equal(actual, expected_arr):
            print(f"FAIL: string {i}. Expected {expected_arr}, got {actual}")
            success = False
            break

    if success:
        print(f"PASS: n={n}, k={k}")

    return success


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [
        (0, 1), (0, 64),
        (1, 1), (1, 64),
        (100, 1), (100, 32), (100, 64),
        (1000, 1), (1000, 64),
        (1000003, 1), (1000003, 64),
    ]

    for n, k in test_cases:
        result = test_string_prefix(n, k)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
