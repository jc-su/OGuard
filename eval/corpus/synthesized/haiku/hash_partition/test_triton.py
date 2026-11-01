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
    "_hash_partition_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def hash_func(x):
    """Reference hash function."""
    # h(x) = (x * 0x9E3779B97F4A7C15) >> 32
    x = int(x)

    # Use Python's arbitrary precision to avoid overflow
    # Treat as 64-bit unsigned
    if x < 0:
        x = x + (1 << 64)

    magic = 0x9E3779B97F4A7C15
    result = (x * magic)

    # Get bits 32-64 (shift right by 32)
    hashed = (result >> 32) & 0xFFFFFFFF

    return hashed


def test_hash_partition(n: int, p: int) -> bool:
    """Test hash_partition kernel."""
    device = torch.device("cuda:0")

    # Create input keys
    if n == 0:
        keys = torch.zeros(0, dtype=torch.int64, device=device)
    else:
        # Use random keys
        keys_data = np.random.randint(-2**63, 2**63-1, size=n, dtype=np.int64)
        keys = torch.from_numpy(keys_data).to(device)

    # Create output arrays
    part = torch.zeros(n, dtype=torch.int32, device=device)
    counts = torch.zeros(p, dtype=torch.int32, device=device)

    # Run kernel
    run(keys, part, counts, n, p)

    # Synchronize
    torch.cuda.synchronize()

    # Check results
    keys_np = keys.cpu().numpy()
    part_np = part.cpu().numpy()
    counts_np = counts.cpu().numpy()

    # Verify partition indices and counts
    actual_counts = np.zeros(p, dtype=np.int32)

    for i in range(n):
        key = keys_np[i]
        partition = part_np[i]

        # Compute expected partition
        h = hash_func(key)
        expected_partition = h % p

        if partition != expected_partition:
            print(f"FAIL: key[{i}]={key}. Expected partition {expected_partition}, got {partition}")
            return False

        # Track counts
        actual_counts[partition] += 1

    # Verify counts
    if not np.array_equal(actual_counts, counts_np):
        print(f"FAIL: n={n}, p={p}")
        print(f"  Expected counts: {actual_counts}")
        print(f"  Got counts:      {counts_np}")
        return False

    print(f"PASS: n={n}, p={p}")
    return True


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [
        (0, 1), (0, 100),
        (1, 1), (1, 10),
        (100, 1), (100, 10), (100, 100),
        (1000, 100),
        (10000, 256),
        (1000003, 128),  # Large n, medium p
    ]

    for n, p in test_cases:
        result = test_hash_partition(n, p)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
