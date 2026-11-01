import numpy as np
import sys
import os
import importlib.util

# First, import the real numba package
from numba import cuda

# Now load the local implementation module with a different name
spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/group_max/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(group, value, n, capacity):
    """CPU reference implementation"""
    best = np.full(capacity, -(2**63), dtype=np.int64)
    for i in range(n):
        g = group[i]
        if 0 <= g < capacity:
            best[g] = max(best[g], value[i])
    return best

def test_group_max(n):
    """Test group_max kernel"""
    capacity = 100

    if n == 0:
        group = np.array([], dtype=np.int32)
        value = np.array([], dtype=np.int64)
    else:
        # Create test data
        group = np.random.randint(0, capacity, n, dtype=np.int32)
        value = np.random.randint(-1000, 1000, n, dtype=np.int64)

    best = np.full(capacity, -(2**63), dtype=np.int64)

    # Run kernel
    try:
        cuda.select_device(0)
        run(group, value, best, n, capacity)

        # Get reference
        ref = cpu_reference(group, value, n, capacity)

        # Compare
        if np.array_equal(best, ref):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            print(f"Expected: {ref}")
            print(f"Got: {best}")
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
        if test_group_max(n):
            passed += 1

    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
