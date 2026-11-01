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
    "_transpose_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def test_transpose(rows: int, cols: int) -> bool:
    """Test transpose kernel."""
    device = torch.device("cuda:0")

    # Create source matrix
    if rows * cols == 0:
        source = torch.zeros(0, dtype=torch.int32, device=device)
    else:
        source_data = np.arange(rows * cols, dtype=np.int32)
        source = torch.from_numpy(source_data).to(device)

    # Create output matrix
    out = torch.zeros(rows * cols, dtype=torch.int32, device=device)

    # Run kernel
    run(source, out, rows, cols)

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    source_np = source.cpu().numpy().reshape(rows, cols) if rows > 0 and cols > 0 else np.zeros((rows, cols), dtype=np.int32)
    out_np = out.cpu().numpy().reshape(cols, rows) if rows > 0 and cols > 0 else np.zeros((cols, rows), dtype=np.int32)

    # Compute expected transpose
    if rows > 0 and cols > 0:
        expected = source_np.T
        if not np.array_equal(out_np, expected):
            print(f"FAIL: rows={rows}, cols={cols}")
            print(f"  Source:\n{source_np}")
            print(f"  Expected:\n{expected}")
            print(f"  Got:\n{out_np}")
            return False

    print(f"PASS: rows={rows}, cols={cols}")
    return True


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [
        (0, 0), (0, 1), (1, 0),
        (1, 1), (2, 3), (3, 2),
        (10, 10), (100, 100),
        (1000, 1000),
        (1003, 999),  # Non-square
        (1000, 1003),  # Non-square
    ]

    for rows, cols in test_cases:
        result = test_transpose(rows, cols)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
