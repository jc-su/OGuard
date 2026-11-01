import numpy as np
import sys
import os
import importlib.util

# First, import the real numba package
from numba import cuda

# Now load the local implementation module with a different name
spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/run_starts/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(values, n):
    """CPU reference implementation"""
    flags = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if i == 0 or values[i] != values[i - 1]:
            flags[i] = 1
    return flags

def test_run_starts(n):
    """Test run_starts kernel"""
    # Create test data
    values = np.random.randint(-1000, 1000, n, dtype=np.int64)
    flags = np.zeros(n, dtype=np.int8)

    # Run kernel
    try:
        cuda.select_device(0)
        run(values, flags, n)

        # Get reference
        ref = cpu_reference(values, n)

        # Compare
        if np.array_equal(flags, ref):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            print(f"Expected: {ref}")
            print(f"Got: {flags}")
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
        if test_run_starts(n):
            passed += 1

    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
