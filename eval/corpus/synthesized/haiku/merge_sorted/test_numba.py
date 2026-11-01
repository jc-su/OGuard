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

def cpu_reference(a_np, b_np, m, p):
    """Compute reference on CPU."""
    # Simple merge
    result = np.zeros(m + p, dtype=np.int64)
    i, j = 0, 0

    for k in range(m + p):
        if i >= m:
            result[k] = b_np[j]
            j += 1
        elif j >= p:
            result[k] = a_np[i]
            i += 1
        elif a_np[i] <= b_np[j]:
            result[k] = a_np[i]
            i += 1
        else:
            result[k] = b_np[j]
            j += 1

    return result

def test_merge_sorted():
    """Test merge_sorted kernel."""
    for m in [0, 1, 1000]:
        for p in [0, 1, 1000]:
            if m + p == 0:
                continue

            # Create test data
            a = np.sort(np.random.randint(-1000, 1000, size=m)).astype(np.int64)
            b = np.sort(np.random.randint(-1000, 1000, size=p)).astype(np.int64)
            out = np.zeros(m + p, dtype=np.int64)

            # Run kernel
            out = run(a, b, out, m, p)

            # Get reference
            ref = cpu_reference(a, b, m, p)

            # Check results
            np.testing.assert_array_equal(out, ref)

            print(f"✓ m={m}, p={p}: passed")

if __name__ == "__main__":
    test_merge_sorted()
    print("All tests passed!")
