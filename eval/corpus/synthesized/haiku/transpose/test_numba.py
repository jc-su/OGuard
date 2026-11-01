import numpy as np
from numba import cuda
import importlib.util

spec = importlib.util.spec_from_file_location("transpose_kernel", "$HOME/Dev/opgate-heldout4/haiku/transpose/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(source, rows, cols):
    """Compute reference output on CPU."""
    out_ref = np.zeros(rows * cols, dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            out_ref[j * rows + i] = source[i * cols + j]
    return out_ref


def test_transpose(rows, cols):
    """Test transpose for given matrix dimensions."""
    # Create input data
    source_np = np.arange(rows * cols, dtype=np.int32)
    out_np = np.zeros(rows * cols, dtype=np.int32)

    # Copy to device
    source_d = cuda.to_device(source_np)
    out_d = cuda.to_device(out_np)

    # Run kernel
    module.run(source_d, out_d, rows, cols)

    # Copy results back
    out_result = out_d.copy_to_host()

    # Compute reference
    ref = cpu_reference(source_np, rows, cols)

    # Check results
    assert np.array_equal(out_result, ref), f"Values mismatch for rows={rows}, cols={cols}"

    print(f"Test passed for rows={rows}, cols={cols}")


if __name__ == "__main__":
    test_cases = [(0, 0), (1, 1), (10, 100), (1000, 1001), (32, 32)]
    for rows, cols in test_cases:
        test_transpose(rows, cols)
    print("All tests passed!")
