import numpy as np
import sys
import os
import importlib.util

# First, import the real numba package
from numba import cuda

# Now load the local implementation module with a different name
spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/bucket_histogram/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(values, bounds, n, b):
    """CPU reference implementation"""
    counts = np.zeros(b + 1, dtype=np.int32)
    for i in range(n):
        val = values[i]
        # Count how many bounds <= val
        count = 0
        for j in range(b):
            if bounds[j] <= val:
                count += 1
        counts[count] += 1
    return counts

def test_bucket_histogram(n):
    """Test bucket_histogram kernel"""
    b = 10

    if n == 0:
        values = np.array([], dtype=np.float64)
        bounds = np.linspace(0, 10, b, dtype=np.float64)
    else:
        # Create test data
        values = np.random.uniform(-5, 15, n).astype(np.float64)
        bounds = np.linspace(0, 10, b, dtype=np.float64)

    counts = np.zeros(b + 1, dtype=np.int32)

    # Run kernel
    try:
        cuda.select_device(0)
        run(values, bounds, counts, n, b)

        # Get reference
        ref = cpu_reference(values, bounds, n, b)

        # Compare
        if np.array_equal(counts, ref):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            print(f"Expected: {ref}")
            print(f"Got: {counts}")
            return False
    except Exception as e:
        import traceback
        print(f"n={n}: ERROR - {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    passed = 0
    total = 0

    for n in [0, 1, 1000, 1000003]:
        total += 1
        if test_bucket_histogram(n):
            passed += 1

    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
