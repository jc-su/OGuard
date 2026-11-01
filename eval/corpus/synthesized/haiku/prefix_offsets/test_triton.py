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
    "_prefix_offsets_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def test_prefix_offsets(n: int) -> bool:
    """Test prefix_offsets kernel."""
    device = torch.device("cuda:0")

    # Create lengths array with small values
    if n == 0:
        lengths = torch.zeros(0, dtype=torch.int32, device=device)
    else:
        # Use values in [0, 2^20]
        lengths_data = np.random.randint(0, 100, size=n, dtype=np.int32)
        lengths = torch.from_numpy(lengths_data).to(device)

    # Create output offsets array
    offsets = torch.zeros(n + 1, dtype=torch.int32, device=device)

    # Run kernel
    run(lengths, offsets, n)

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    lengths_np = lengths.cpu().numpy()
    offsets_np = offsets.cpu().numpy()

    # Compute expected offsets
    expected = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        expected[i + 1] = expected[i] + lengths_np[i]

    if not np.array_equal(offsets_np, expected):
        print(f"FAIL: n={n}")
        print(f"  Expected: {expected[:min(10, len(expected))]}")
        print(f"  Got:      {offsets_np[:min(10, len(offsets_np))]}")
        return False

    print(f"PASS: n={n}, total_length={offsets_np[-1]}")
    return True


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [0, 1, 10, 100, 1000, 10000, 100000, 1000003]

    for n in test_cases:
        result = test_prefix_offsets(n)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
