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

def cpu_reference(offsets_np, chars_np, pattern_np, n, k):
    """Compute reference on CPU."""
    match = np.zeros(n, dtype=np.int8)
    if k == 0:
        match[:] = 1
    else:
        for i in range(n):
            start = offsets_np[i]
            end = offsets_np[i + 1]
            string = bytes(chars_np[start:end])
            pattern_bytes = bytes(pattern_np[:k])
            match[i] = 1 if pattern_bytes in string else 0
    return match

def test_string_contains():
    """Test string_contains kernel."""
    for n in [0, 1, 1000, 1000003]:
        # Create test data
        if n == 0:
            offsets = np.array([0], dtype=np.int32)
            chars = np.array([], dtype=np.uint8)
            pattern = np.array([1, 2, 3], dtype=np.uint8)
            k = 3
        else:
            # Create random strings
            total_bytes = n * 10  # Average 10 bytes per string
            offsets = np.zeros(n + 1, dtype=np.int32)
            offsets[0] = 0
            for i in range(n):
                str_len = np.random.randint(1, 21)
                offsets[i + 1] = offsets[i] + str_len

            chars = np.random.randint(0, 256, size=offsets[n], dtype=np.uint8)
            k = np.random.randint(0, 5)
            pattern = np.random.randint(0, 256, size=k, dtype=np.uint8)

        match = np.zeros(n, dtype=np.int8)

        # Run kernel
        match = run(offsets, chars, pattern, match, n, k)

        # Get reference
        ref = cpu_reference(offsets, chars, pattern, n, k)

        # Check results
        np.testing.assert_array_equal(match, ref)

        print(f"✓ n={n}, k={k}: passed")

if __name__ == "__main__":
    test_string_contains()
    print("All tests passed!")
