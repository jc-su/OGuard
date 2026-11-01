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

def cpu_reference(values_np, n, window):
    """Compute reference on CPU."""
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.sum(values_np[start:i + 1])
    return out

def test_rolling_sum():
    """Test rolling_sum kernel."""
    for n in [0, 1, 1000, 1000003]:
        for window in [1, 5, 100]:
            if window > n and n > 0:
                continue

            # Create test data
            values = np.random.randn(n).astype(np.float64)
            out = np.zeros(n, dtype=np.float64)

            # Run kernel
            out = run(values, out, n, window)

            # Get reference
            ref = cpu_reference(values, n, window)

            # Check results (allow for floating point precision)
            np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-14)

            print(f"✓ n={n}, window={window}: passed")

if __name__ == "__main__":
    test_rolling_sum()
    print("All tests passed!")
