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
    "_pair_sum_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def test_pair_sum(m: int, l: int, r: int) -> bool:
    """Test pair_sum kernel."""
    device = torch.device("cuda:0")

    # Create input arrays
    if l == 0:
        left = torch.zeros(0, dtype=torch.float64, device=device)
    else:
        left_data = np.random.randn(l)
        left = torch.from_numpy(left_data).to(device)

    if r == 0:
        right = torch.zeros(0, dtype=torch.float64, device=device)
    else:
        right_data = np.random.randn(r)
        right = torch.from_numpy(right_data).to(device)

    # Create index arrays
    if m == 0:
        left_idx = torch.zeros(0, dtype=torch.int32, device=device)
        right_idx = torch.zeros(0, dtype=torch.int32, device=device)
    else:
        left_idx_data = np.random.randint(0, max(1, l), size=m, dtype=np.int32)
        right_idx_data = np.random.randint(0, max(1, r), size=m, dtype=np.int32)
        left_idx = torch.from_numpy(left_idx_data).to(device)
        right_idx = torch.from_numpy(right_idx_data).to(device)

    # Create output array
    out = torch.zeros(m, dtype=torch.float64, device=device)

    # Run kernel
    run(left_idx, right_idx, left, right, out, m, l, r)

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    left_np = left.cpu().numpy()
    right_np = right.cpu().numpy()
    left_idx_np = left_idx.cpu().numpy()
    right_idx_np = right_idx.cpu().numpy()
    out_np = out.cpu().numpy()

    # Verify results
    for i in range(m):
        li = int(left_idx_np[i])
        ri = int(right_idx_np[i])

        expected = left_np[li] + right_np[ri]
        actual = out_np[i]

        if not np.isclose(actual, expected):
            print(f"FAIL: pair[{i}]. left[{li}]={left_np[li]}, right[{ri}]={right_np[ri]}")
            print(f"  Expected: {expected}")
            print(f"  Got: {actual}")
            return False

    print(f"PASS: m={m}, l={l}, r={r}")
    return True


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [
        (0, 0, 0),
        (1, 1, 1),
        (10, 10, 10),
        (100, 50, 50),
        (1000, 100, 100),
        (10000, 1000, 1000),
        (1000003, 10000, 10000),
    ]

    for m, l, r in test_cases:
        result = test_pair_sum(m, l, r)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
