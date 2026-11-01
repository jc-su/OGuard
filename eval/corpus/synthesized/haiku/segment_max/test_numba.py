import numpy as np
import sys
import os
import importlib.util

# First, import the real numba package
from numba import cuda

# Now load the local implementation module with a different name
spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/segment_max/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(offsets, values, g):
    """CPU reference implementation"""
    maxes = np.full(g, -(2**63), dtype=np.int64)
    for i in range(g):
        start = offsets[i]
        end = offsets[i + 1]
        if start < end:
            maxes[i] = np.max(values[start:end])
        else:
            maxes[i] = -(2**63)
    return maxes

def test_segment_max(n):
    """Test segment_max kernel"""
    if n == 0:
        g = 0
        offsets = np.array([0], dtype=np.int32)
        values = np.array([], dtype=np.int64)
        maxes = np.zeros(g, dtype=np.int64)
    else:
        # Create test data - generate segments
        g = min(n // 10 + 1, 100)  # Number of segments
        values = np.random.randint(-1000, 1000, n, dtype=np.int64)

        # Create offsets for segments
        offsets = np.sort(np.random.choice(n + 1, g, replace=False)).astype(np.int32)
        offsets = np.append(offsets, np.int32(n))
        offsets = np.unique(offsets)
        g = len(offsets) - 1

        maxes = np.zeros(g, dtype=np.int64)

    # Run kernel
    try:
        cuda.select_device(0)
        run(offsets, values, maxes, g)

        # Get reference
        ref = cpu_reference(offsets, values, g)

        # Compare
        if np.array_equal(maxes, ref):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            print(f"Expected: {ref}")
            print(f"Got: {maxes}")
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
        if test_segment_max(n):
            passed += 1

    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
