import numpy as np
from numba import cuda
from datetime import datetime, timedelta
import importlib.util

spec = importlib.util.spec_from_file_location("month_start_kernel", "$HOME/Dev/opgate-heldout4/haiku/month_start/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def day_number_from_date(year, month, day):
    """Convert a date to day number since 1970-01-01."""
    d = datetime(year, month, day)
    epoch = datetime(1970, 1, 1)
    return (d - epoch).days


def cpu_reference(days, n):
    """Compute reference output on CPU."""
    out_ref = np.zeros(n, dtype=np.int32)
    epoch = datetime(1970, 1, 1)

    for i in range(n):
        # Convert day number to date
        day_date = epoch + timedelta(days=int(days[i]))
        # Get the first day of that month
        month_start = datetime(day_date.year, day_date.month, 1)
        # Convert back to day number
        out_ref[i] = (month_start - epoch).days

    return out_ref


def test_month_start(n):
    """Test month_start for given size."""
    # Create input data
    days_np = np.array([10 + i * 3 for i in range(n)], dtype=np.int32)
    out_np = np.zeros(n, dtype=np.int32)

    # Copy to device
    days_d = cuda.to_device(days_np)
    out_d = cuda.to_device(out_np)

    # Run kernel
    module.run(days_d, out_d, n)

    # Copy results back
    out_result = out_d.copy_to_host()

    # Compute reference
    ref = cpu_reference(days_np, n)

    # Check results
    assert np.array_equal(out_result, ref), f"Values mismatch for n={n}"

    print(f"Test passed for n={n}")


if __name__ == "__main__":
    for n in [0, 1, 1000, 10000]:
        test_month_start(n)
    print("All tests passed!")
