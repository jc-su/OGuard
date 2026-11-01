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

def cpu_reference(valid_np, n):
    """Compute reference on CPU."""
    valid_rows = []
    for i in range(n):
        byte_idx = i // 8
        bit_idx = i % 8
        if byte_idx < len(valid_np):
            is_valid = (valid_np[byte_idx] >> bit_idx) & 1
            if is_valid:
                valid_rows.append(i)
    return np.array(valid_rows, dtype=np.int32), len(valid_rows)

def test_valid_rows():
    """Test valid_rows kernel."""
    for n in [0, 1, 1000, 1000003]:
        # Create test data
        bitmap_size = (n + 7) // 8
        valid = np.random.randint(0, 256, size=bitmap_size, dtype=np.uint8)
        rows = np.zeros(n, dtype=np.int32)
        count = np.array([0], dtype=np.int32)

        # Run kernel
        rows, count_result = run(valid, rows, count, n)

        # Get reference
        ref_rows, ref_count = cpu_reference(valid, n)

        # Check count
        assert count_result[0] == ref_count, f"n={n}: count mismatch: got {count_result[0]}, expected {ref_count}"

        # Check rows (order doesn't matter, so sort both)
        got_rows = sorted(rows[:count_result[0]].tolist())
        expected_rows = sorted(ref_rows.tolist())
        assert got_rows == expected_rows, f"n={n}: rows mismatch"

        print(f"✓ n={n}: passed (count={count_result[0]})")

if __name__ == "__main__":
    test_valid_rows()
    print("All tests passed!")
