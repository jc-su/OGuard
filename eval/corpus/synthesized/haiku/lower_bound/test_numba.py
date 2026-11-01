import numpy as np
import sys
import os
import importlib.util

# First, import the real numba package
from numba import cuda

# Now load the local implementation module with a different name
spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/lower_bound/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(keys, sorted_arr, n, s):
    """CPU reference implementation"""
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        key = keys[i]
        # Count elements < key
        count = 0
        for j in range(s):
            if sorted_arr[j] < key:
                count += 1
            else:
                break
        out[i] = count
    return out

def test_lower_bound(n):
    """Test lower_bound kernel"""
    if n == 0:
        keys = np.array([], dtype=np.int64)
        sorted_arr = np.array([1, 2, 3, 5, 8, 13], dtype=np.int64)
        out = np.zeros(n, dtype=np.int32)
        s = 6
    else:
        # Create sorted array
        sorted_arr = np.sort(np.random.randint(-1000, 1000, 100, dtype=np.int64))
        s = len(sorted_arr)

        # Create keys
        keys = np.random.randint(-1500, 1500, n, dtype=np.int64)
        out = np.zeros(n, dtype=np.int32)

    # Run kernel
    try:
        cuda.select_device(0)
        run(keys, sorted_arr, out, n, s)

        # Get reference
        ref = cpu_reference(keys, sorted_arr, n, s)

        # Compare
        if np.array_equal(out, ref):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            print(f"Expected: {ref}")
            print(f"Got: {out}")
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
        if test_lower_bound(n):
            passed += 1

    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
