import numpy as np
from numba import cuda
import importlib.util

spec = importlib.util.spec_from_file_location("window_max_kernel", "$HOME/Dev/opgate-heldout4/haiku/window_max/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(values, n, w):
    """Compute reference output on CPU."""
    out_ref = np.zeros(n, dtype=np.int64)
    for i in range(n):
        start = max(0, i - w + 1)
        end = i + 1
        out_ref[i] = np.max(values[start:end])
    return out_ref


def test_window_max(n, w):
    """Test window_max for given size."""
    # Create input data
    values_np = np.random.randint(-1000, 1000, n, dtype=np.int64)
    out_np = np.zeros(n, dtype=np.int64)

    # Copy to device
    values_d = cuda.to_device(values_np)
    out_d = cuda.to_device(out_np)

    # Run kernel
    module.run(values_d, out_d, n, w)

    # Copy results back
    out_result = out_d.copy_to_host()

    # Compute reference
    ref = cpu_reference(values_np, n, w)

    # Check results
    assert np.array_equal(out_result, ref), f"Values mismatch for n={n}, w={w}"

    print(f"Test passed for n={n}, w={w}")


if __name__ == "__main__":
    test_cases = [(0, 1), (1, 1), (100, 5), (1000, 32), (1000003, 256)]
    for n, w in test_cases:
        test_window_max(n, w)
    print("All tests passed!")
