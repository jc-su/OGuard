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
    "_window_max_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def test_window_max(n: int, w: int) -> bool:
    """Test window_max kernel."""
    device = torch.device("cuda:0")

    # Create input values
    if n == 0:
        values = torch.zeros(0, dtype=torch.int64, device=device)
    else:
        # Use random values
        values_data = np.random.randint(-1000, 1000, size=n, dtype=np.int64)
        values = torch.from_numpy(values_data).to(device)

    # Create output array
    out = torch.zeros(n, dtype=torch.int64, device=device)

    # Run kernel
    run(values, out, n, w)

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    values_np = values.cpu().numpy()
    out_np = out.cpu().numpy()

    # Verify results
    for i in range(n):
        start = max(0, i - w + 1)
        window = values_np[start:i+1]
        expected = np.max(window)
        actual = int(out_np[i])

        if actual != expected:
            print(f"FAIL: i={i}, w={w}")
            print(f"  Window: {window}")
            print(f"  Expected: {expected}")
            print(f"  Got: {actual}")
            return False

    print(f"PASS: n={n}, w={w}")
    return True


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [
        (0, 1),
        (1, 1),
        (10, 1), (10, 3), (10, 10), (10, 100),
        (100, 1), (100, 10), (100, 100),
        (1000, 1), (1000, 100),
        (1000003, 1), (1000003, 1024),
    ]

    for n, w in test_cases:
        result = test_window_max(n, w)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
