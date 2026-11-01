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

def cpu_reference(run_values_np, run_ends_np, n):
    """Compute reference on CPU."""
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Find which run row i belongs to
        run_idx = np.searchsorted(run_ends_np, i + 1, side='left')
        out[i] = run_values_np[run_idx]
    return out

def test_run_end_decode():
    """Test run_end_decode kernel."""
    for n in [0, 1, 1000, 1000003]:
        r = max(1, n // 100 + 1)

        # Create test data
        run_values = np.random.randn(r).astype(np.float64)
        # Create strictly increasing run_ends
        if n == 0:
            run_ends = np.array([0], dtype=np.int32)
        else:
            run_ends_unsorted = np.sort(np.random.choice(range(1, n + 1), size=r - 1, replace=False))
            run_ends = np.concatenate([run_ends_unsorted, np.array([n], dtype=np.int32)])
            run_ends = run_ends.astype(np.int32)

        out = np.zeros(n, dtype=np.float64)

        # Run kernel
        out = run(run_values, run_ends, out, r, n)

        # Get reference
        ref = cpu_reference(run_values, run_ends, n)

        # Check results
        np.testing.assert_allclose(out, ref, rtol=1e-10)

        print(f"✓ n={n}, r={r}: passed")

if __name__ == "__main__":
    test_run_end_decode()
    print("All tests passed!")
