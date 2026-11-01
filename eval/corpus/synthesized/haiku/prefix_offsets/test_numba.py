import numpy as np
from numba import cuda
import importlib.util

spec = importlib.util.spec_from_file_location("prefix_offsets_kernel", "$HOME/Dev/opgate-heldout4/haiku/prefix_offsets/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(lengths, n):
    """Compute reference output on CPU."""
    offsets_ref = np.zeros(n + 1, dtype=np.int32)
    total = 0
    for i in range(n):
        offsets_ref[i] = total
        total += lengths[i]
    offsets_ref[n] = total
    return offsets_ref


def test_prefix_offsets(n):
    """Test prefix_offsets for given size."""
    # Create input data
    lengths_np = np.random.randint(0, 1000, n, dtype=np.int32)
    offsets_np = np.zeros(n + 1, dtype=np.int32)

    # Copy to device
    lengths_d = cuda.to_device(lengths_np)
    offsets_d = cuda.to_device(offsets_np)

    # Run kernel
    module.run(lengths_d, offsets_d, n)

    # Copy results back
    offsets_result = offsets_d.copy_to_host()

    # Compute reference
    ref = cpu_reference(lengths_np, n)

    # Check results
    assert np.array_equal(offsets_result, ref), f"Values mismatch for n={n}"

    print(f"Test passed for n={n}")


if __name__ == "__main__":
    for n in [0, 1, 1000, 1000003]:
        test_prefix_offsets(n)
    print("All tests passed!")
