import numpy as np
import sys
import os
import importlib.util

# First, import the real numba package
from numba import cuda

# Now load the local implementation module with a different name
spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/masked_sum/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(values, mask, n):
    """CPU reference implementation"""
    result = 0.0
    for i in range(n):
        if mask[i] != 0:
            result += values[i]
    return result

def test_masked_sum(n):
    """Test masked_sum kernel"""
    if n == 0:
        values = np.array([], dtype=np.float64)
        mask = np.array([], dtype=np.int8)
    else:
        # Create test data
        values = np.random.randn(n).astype(np.float64)
        mask = np.random.randint(-10, 10, n, dtype=np.int8)

    result = np.array([0.0], dtype=np.float64)

    # Run kernel
    try:
        cuda.select_device(0)
        run(values, mask, result, n)

        # Get reference
        ref = cpu_reference(values, mask, n)

        # Compare (use allclose for floating point)
        if np.allclose(result[0], ref, rtol=1e-9):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            print(f"Expected: {ref}")
            print(f"Got: {result[0]}")
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
        if test_masked_sum(n):
            passed += 1

    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
