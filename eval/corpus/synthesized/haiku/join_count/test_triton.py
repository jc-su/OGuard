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
    "_join_count_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def test_join_count(b: int, p: int) -> bool:
    """Test join_count kernel."""
    device = torch.device("cuda:0")

    # Create sorted build array
    if b == 0:
        build = torch.tensor([], dtype=torch.int64, device=device)
    else:
        # Create a sorted array with some duplicates
        build_data = []
        val = 0
        while len(build_data) < b:
            build_data.extend([val] * (val % 3 + 1))
            val += 1
        build = torch.tensor(sorted(build_data)[:b], dtype=torch.int64, device=device)

    # Create probe array
    if p == 0:
        probe = torch.tensor([], dtype=torch.int64, device=device)
    else:
        # Use values that should have matches
        probe_data = np.random.randint(0, 20, size=p)
        probe = torch.from_numpy(probe_data.astype(np.int64)).to(device)

    # Create output array
    counts = torch.zeros(p, dtype=torch.int32, device=device)

    # Run kernel
    run(build, probe, counts, b, p)

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    build_np = build.cpu().numpy()
    probe_np = probe.cpu().numpy()
    counts_np = counts.cpu().numpy()

    success = True
    for i in range(p):
        probe_val = probe_np[i]
        # Count expected matches in build
        expected = np.sum(build_np == probe_val)
        actual = int(counts_np[i])

        if actual != expected:
            print(f"FAIL: probe[{i}]={probe_val}. Expected {expected}, got {actual}")
            success = False

    if success:
        print(f"PASS: b={b}, p={p}")

    return success


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (100, 100),
        (1000, 1000),
        (1000003, 100),  # Large build, small probe
        (100, 1000003),  # Small build, large probe
    ]

    for b, p in test_cases:
        result = test_join_count(b, p)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
