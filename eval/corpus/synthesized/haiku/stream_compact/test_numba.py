import numpy as np
from numba import cuda

# Import the numba module
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("stream_compact_kernel", "$HOME/Dev/opgate-heldout4/haiku/stream_compact/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(values, keep, n):
    """Compute reference output on CPU."""
    mask = keep[:n] != 0
    return values[:n][mask]


def test_stream_compact(n):
    """Test stream_compact for a given size."""
    # Create input data
    values_np = np.arange(n, dtype=np.int64)
    keep_np = np.random.randint(0, 2, n, dtype=np.int8)
    out_np = np.zeros(n, dtype=np.int64)
    count_np = np.zeros(1, dtype=np.int32)

    # Copy to device
    values_d = cuda.to_device(values_np)
    keep_d = cuda.to_device(keep_np)
    out_d = cuda.to_device(out_np)
    count_d = cuda.to_device(count_np)

    # Run kernel
    module.run(values_d, keep_d, out_d, count_d, n)

    # Copy results back
    out_result = out_d.copy_to_host()
    count_result = count_d.copy_to_host()[0]

    # Compute reference
    ref = cpu_reference(values_np, keep_np, n)

    # Check count
    assert count_result == len(ref), f"Count mismatch: got {count_result}, expected {len(ref)}"

    # Check values (in any order, so we sort both)
    out_sorted = np.sort(out_result[:count_result])
    ref_sorted = np.sort(ref)
    assert np.array_equal(out_sorted, ref_sorted), f"Values mismatch for n={n}"

    print(f"Test passed for n={n}: {count_result} elements")


if __name__ == "__main__":
    for n in [0, 1, 1000, 1000003]:
        test_stream_compact(n)
    print("All tests passed!")
