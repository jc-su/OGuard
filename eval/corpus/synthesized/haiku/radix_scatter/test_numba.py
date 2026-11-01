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

def cpu_reference(keys_np, cursors_np_init, n, shift):
    """Compute reference on CPU."""
    cursors = cursors_np_init.copy()
    out = np.zeros(n, dtype=np.int32)

    for i in range(n):
        byte_val = (keys_np[i] >> shift) & 0xFF
        pos = cursors[byte_val]
        out[pos] = i
        cursors[byte_val] += 1

    return out, cursors

def test_radix_scatter():
    """Test radix_scatter kernel."""
    for n in [0, 1, 1000, 1000003]:
        for shift in [0, 8, 16, 24]:
            # Create test data
            keys = np.random.randint(0, 2**32, size=n, dtype=np.uint32)

            # Compute initial cursor positions (exclusive prefix sum)
            if n == 0:
                cursors_init = np.zeros(256, dtype=np.int32)
            else:
                counts = np.bincount([(k >> shift) & 0xFF for k in keys], minlength=256)
                cursors_init = np.cumsum(np.concatenate([[0], counts[:-1]])).astype(np.int32)

            cursors = cursors_init.copy()
            out = np.zeros(n, dtype=np.int32)

            # Run kernel
            out, cursors = run(keys, cursors, out, n, shift)

            # Get reference
            ref_out, ref_cursors = cpu_reference(keys, cursors_init, n, shift)

            # Check results
            # Verify grouped by byte (order within group doesn't matter per spec)
            for byte_val in range(256):
                mask = ((keys >> shift) & 0xFF) == byte_val
                expected_rows = np.where(mask)[0]
                got_rows = out[cursors_init[byte_val]:cursors[byte_val]]
                np.testing.assert_array_equal(sorted(got_rows), sorted(expected_rows))
            np.testing.assert_array_equal(cursors, ref_cursors)

            print(f"✓ n={n}, shift={shift}: passed")

if __name__ == "__main__":
    test_radix_scatter()
    print("All tests passed!")
