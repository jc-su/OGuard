import numpy as np
from numba import cuda
import importlib.util

spec = importlib.util.spec_from_file_location("pair_sum_kernel", "$HOME/Dev/opgate-heldout4/haiku/pair_sum/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(left_idx, right_idx, left, right, m, l, r):
    """Compute reference output on CPU."""
    out_ref = np.zeros(m, dtype=np.float64)
    for i in range(m):
        out_ref[i] = left[left_idx[i]] + right[right_idx[i]]
    return out_ref


def test_pair_sum(m, l, r):
    """Test pair_sum for given sizes."""
    # Create input data
    left_idx_np = np.random.randint(0, l, m, dtype=np.int32)
    right_idx_np = np.random.randint(0, r, m, dtype=np.int32)
    left_np = np.random.randn(l).astype(np.float64)
    right_np = np.random.randn(r).astype(np.float64)
    out_np = np.zeros(m, dtype=np.float64)

    # Copy to device
    left_idx_d = cuda.to_device(left_idx_np)
    right_idx_d = cuda.to_device(right_idx_np)
    left_d = cuda.to_device(left_np)
    right_d = cuda.to_device(right_np)
    out_d = cuda.to_device(out_np)

    # Run kernel
    module.run(left_idx_d, right_idx_d, left_d, right_d, out_d, m, l, r)

    # Copy results back
    out_result = out_d.copy_to_host()

    # Compute reference
    ref = cpu_reference(left_idx_np, right_idx_np, left_np, right_np, m, l, r)

    # Check results (using approximate comparison for floats)
    assert np.allclose(out_result, ref, rtol=1e-10), f"Values mismatch for m={m}, l={l}, r={r}"

    print(f"Test passed for m={m}, l={l}, r={r}")


if __name__ == "__main__":
    test_cases = [(0, 0, 0), (1, 1, 1), (100, 50, 50), (1000, 100, 100), (1000003, 10000, 10000)]
    for m, l, r in test_cases:
        if m > 0 and l > 0 and r > 0:  # Skip empty cases for now
            test_pair_sum(m, l, r)
    print("All tests passed!")
