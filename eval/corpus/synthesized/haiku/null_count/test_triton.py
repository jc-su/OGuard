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
    "_null_count_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def test_null_count(n: int) -> bool:
    """Test null_count kernel."""
    device = torch.device("cuda:0")

    # Create validity bitmap
    num_bytes = (n + 7) >> 3

    if num_bytes == 0:
        valid = torch.zeros(0, dtype=torch.uint8, device=device)
    else:
        # Create a random bitmap
        valid_data = np.random.randint(0, 256, size=num_bytes, dtype=np.uint8)
        valid = torch.from_numpy(valid_data).to(device)

    # Create output counter
    count = torch.zeros(1, dtype=torch.int64, device=device)

    # Run kernel
    run(valid, count, n)

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    valid_np = valid.cpu().numpy()
    count_val = int(count[0].item())

    # Compute expected null count
    expected_null_count = 0
    for byte_idx in range(num_bytes):
        byte_val = valid_np[byte_idx] if byte_idx < len(valid_np) else 0

        # How many rows does this byte contain?
        start_row = byte_idx * 8
        end_row = min(start_row + 8, n)
        rows_in_byte = end_row - start_row

        # Count null bits (0s)
        for bit_idx in range(rows_in_byte):
            bit_val = (byte_val >> bit_idx) & 1
            if bit_val == 0:
                expected_null_count += 1

    if count_val != expected_null_count:
        print(f"FAIL: n={n}. Expected {expected_null_count}, got {count_val}")
        return False

    print(f"PASS: n={n}, null_count={count_val}")
    return True


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [0, 1, 7, 8, 9, 15, 16, 17, 100, 1000, 1000003]

    for n in test_cases:
        result = test_null_count(n)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
