import numpy as np
import sys
import os
import importlib.util

# First, import the real numba package
from numba import cuda

# Now load the local implementation module with a different name
spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/string_reverse/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(offsets, chars, n):
    """CPU reference implementation"""
    total_chars = offsets[n]
    out = np.zeros(total_chars, dtype=np.uint8)
    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        length = end - start
        for j in range(length):
            out[start + j] = chars[end - 1 - j]
    return out

def test_string_reverse(n):
    """Test string_reverse kernel"""
    if n == 0:
        offsets = np.array([0], dtype=np.int32)
        chars = np.array([], dtype=np.uint8)
        out = np.array([], dtype=np.uint8)
    else:
        # Create test data with random string offsets
        offsets = np.array([0], dtype=np.int32)
        total_chars = 0
        for _ in range(n):
            str_len = np.random.randint(1, 100)
            total_chars += str_len
            offsets = np.append(offsets, np.int32(total_chars))

        # Create random chars
        chars = np.random.randint(65, 91, total_chars, dtype=np.uint8)  # A-Z
        out = np.zeros(total_chars, dtype=np.uint8)

    # Run kernel
    try:
        cuda.select_device(0)
        run(offsets, chars, out, n)

        # Get reference
        ref = cpu_reference(offsets, chars, n)

        # Compare
        if np.array_equal(out, ref):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            print(f"Expected: {ref[:20]}")
            print(f"Got: {out[:20]}")
            return False
    except Exception as e:
        import traceback
        print(f"n={n}: ERROR - {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    passed = 0
    total = 0

    for n in [0, 1, 1000, 10000]:
        total += 1
        if test_string_reverse(n):
            passed += 1

    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
