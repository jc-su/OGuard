import numpy as np
import sys
from pathlib import Path
import importlib.util

# Load numba.py by path
numba_path = Path(__file__).parent / "numba.py"
spec = importlib.util.spec_from_file_location("_local_numba_kernel", numba_path)
_local_numba_kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_local_numba_kernel)

run = _local_numba_kernel.run

def is_leap(year):
    """Check if year is a leap year."""
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    return False

def cpu_reference(days_np, n):
    """Compute reference on CPU."""
    year = np.zeros(n, dtype=np.int32)

    for i in range(n):
        day_count = days_np[i]

        y = 1970
        remaining = day_count

        if remaining >= 0:
            while True:
                days_in_year = 366 if is_leap(y) else 365
                if remaining < days_in_year:
                    break
                remaining -= days_in_year
                y += 1
        else:
            y -= 1
            remaining = -remaining
            while True:
                days_in_year = 366 if is_leap(y) else 365
                if remaining <= days_in_year:
                    break
                remaining -= days_in_year
                y -= 1

        year[i] = y

    return year

def test_days_to_year():
    """Test days_to_year kernel."""
    import datetime

    for n in [0, 1, 1000, 1000003]:
        # Create test data
        # Use realistic dates
        if n == 0:
            days = np.array([], dtype=np.int32)
        else:
            days = np.random.randint(-21900, 11000, size=n, dtype=np.int32)  # ~1910 to 2000

        year = np.zeros(n, dtype=np.int32)

        # Run kernel
        year = run(days, year, n)

        # Get reference
        ref = cpu_reference(days, n)

        # Check results
        np.testing.assert_array_equal(year, ref)

        print(f"✓ n={n}: passed")

if __name__ == "__main__":
    test_days_to_year()
    print("All tests passed!")
