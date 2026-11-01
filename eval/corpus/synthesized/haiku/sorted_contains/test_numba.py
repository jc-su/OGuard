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

def cpu_reference(keys_np, members_np, n, s):
    """Compute reference on CPU."""
    match = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Binary search
        pos = np.searchsorted(members_np[:s], keys_np[i])
        if pos < s and members_np[pos] == keys_np[i]:
            match[i] = 1
    return match

def test_sorted_contains():
    """Test sorted_contains kernel."""
    for n in [0, 1, 1000, 1000003]:
        for s in [1, 10, 100, 1000]:
            # Create test data
            members = np.sort(np.random.randint(0, 10000, size=s, dtype=np.int64))
            # Remove duplicates
            members = np.unique(members)
            s_actual = len(members)

            # Create keys, some from members, some not
            if n > 0:
                num_present = n // 2
                present_keys = np.random.choice(members, size=num_present)
                absent_keys = np.random.choice(
                    np.setdiff1d(np.arange(10000), members),
                    size=n - num_present
                ).astype(np.int64)
                keys = np.concatenate([present_keys, absent_keys])
                np.random.shuffle(keys)
            else:
                keys = np.array([], dtype=np.int64)

            match = np.zeros(n, dtype=np.int8)

            # Run kernel
            match = run(keys, members, match, n, s_actual)

            # Get reference
            ref = cpu_reference(keys, members, n, s_actual)

            # Check results
            np.testing.assert_array_equal(match, ref)

            print(f"✓ n={n}, s={s_actual}: passed")

if __name__ == "__main__":
    test_sorted_contains()
    print("All tests passed!")
