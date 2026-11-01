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
    "_stream_compact_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def test_stream_compact(n: int) -> bool:
    """Test stream_compact kernel."""
    device = torch.device("cuda:0")

    # Generate test data
    values = torch.arange(n, dtype=torch.int64, device=device)
    # Keep every second element
    keep = torch.zeros(n, dtype=torch.int8, device=device)
    if n > 0:
        keep[::2] = 1

    out = torch.zeros(n, dtype=torch.int64, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)

    # Run kernel
    run(values, keep, out, count, n)

    # Synchronize to ensure kernel is done
    torch.cuda.synchronize()

    # Check results
    count_val = int(count[0].item())
    out_values = out[:count_val].cpu().numpy()

    # Expected: keep every second element
    expected = np.arange(0, n, 2, dtype=np.int64)

    if count_val != len(expected):
        print(f"FAIL: count mismatch. Got {count_val}, expected {len(expected)}")
        return False

    # Check if output contains the correct values (order doesn't matter per spec)
    out_sorted = np.sort(out_values)
    expected_sorted = np.sort(expected)

    if not np.array_equal(out_sorted, expected_sorted):
        print(f"FAIL: values mismatch. Got {out_sorted}, expected {expected_sorted}")
        return False

    print(f"PASS: n={n}, count={count_val}")
    return True


if __name__ == "__main__":
    torch.cuda.set_device(0)  # When CUDA_VISIBLE_DEVICES=1 is set, that device becomes 0

    all_pass = True
    for n in [0, 1, 1000, 1000003]:
        result = test_stream_compact(n)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
