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

def cpu_reference(values_np, n):
    """Compute reference on CPU."""
    if n == 0:
        return np.array([float('inf'), float('-inf')], dtype=np.float64)
    return np.array([np.min(values_np[:n]), np.max(values_np[:n])], dtype=np.float64)

def test_min_max():
    """Test min_max kernel."""
    for n in [0, 1, 1000, 1000003]:
        # Create test data
        values = np.random.randn(n).astype(np.float64) * 1000
        result = np.array([float('inf'), float('-inf')], dtype=np.float64)

        # Run kernel
        result = run(values, result, n)

        # Get reference
        ref = cpu_reference(values, n)

        # Check results
        if n == 0:
            assert result[0] == float('inf') and result[1] == float('-inf')
        else:
            assert result[0] == ref[0], f"min: got {result[0]}, expected {ref[0]}"
            assert result[1] == ref[1], f"max: got {result[1]}, expected {ref[1]}"

        print(f"✓ n={n}: passed (min={result[0]}, max={result[1]})")

if __name__ == "__main__":
    test_min_max()
    print("All tests passed!")
