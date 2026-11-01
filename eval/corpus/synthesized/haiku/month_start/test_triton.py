#!/usr/bin/env python
import sys
import os
from datetime import datetime, timedelta

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
    "_month_start_triton",
    kernel_path,
)
kernel_module = importlib.util.module_from_spec(spec)

# Inject the real triton and tl into the module's namespace before executing
kernel_module.triton = _real_triton
kernel_module.tl = _real_tl
kernel_module.torch = torch

spec.loader.exec_module(kernel_module)

run = kernel_module.run


def day_number_to_date(day_num):
    """Convert day number to datetime."""
    epoch = datetime(1970, 1, 1)
    return epoch + timedelta(days=int(day_num))


def date_to_day_number(dt):
    """Convert datetime to day number."""
    epoch = datetime(1970, 1, 1)
    return (dt - epoch).days


def test_month_start(n: int) -> bool:
    """Test month_start kernel."""
    device = torch.device("cuda:0")

    # Create input day numbers
    if n == 0:
        days = torch.zeros(0, dtype=torch.int32, device=device)
    else:
        # Use random day numbers in the valid range
        # [-719162, 2932896]
        min_day = -719162
        max_day = 2932896
        days_data = np.random.randint(min_day, max_day + 1, size=n, dtype=np.int32)
        days = torch.from_numpy(days_data).to(device)

    # Create output array
    out = torch.zeros(n, dtype=torch.int32, device=device)

    # Run kernel
    try:
        run(days, out, n)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"ERROR during kernel execution: {e}")
        return False

    # Check results
    days_np = days.cpu().numpy()
    out_np = out.cpu().numpy()

    # Verify results (check first few values)
    for i in range(min(n, 10)):
        day_num = int(days_np[i])
        result_day_num = int(out_np[i])

        try:
            # Get the date for the input
            dt = day_number_to_date(day_num)
            # Get the first day of that month
            first_day = datetime(dt.year, dt.month, 1)
            expected_day_num = date_to_day_number(first_day)

            if result_day_num != expected_day_num:
                print(f"FAIL: day[{i}]={day_num} ({dt.date()})")
                print(f"  Expected: {expected_day_num} ({first_day.date()})")
                print(f"  Got: {result_day_num}")
                return False
        except Exception as e:
            print(f"Error checking result {i}: {e}")
            return False

    print(f"PASS: n={n}")
    return True


if __name__ == "__main__":
    torch.cuda.set_device(0)

    all_pass = True

    # Test different sizes
    test_cases = [0, 1, 10, 100, 1000]

    for n in test_cases:
        result = test_month_start(n)
        all_pass = all_pass and result

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
