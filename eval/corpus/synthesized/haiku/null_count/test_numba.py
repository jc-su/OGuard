import numpy as np
from numba import cuda
import importlib.util

spec = importlib.util.spec_from_file_location("null_count_kernel", "$HOME/Dev/opgate-heldout4/haiku/null_count/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(valid, n):
    """Compute reference output on CPU."""
    count = 0
    for i in range(n):
        byte_idx = i // 8
        bit_idx = i % 8
        if byte_idx < len(valid):
            if (valid[byte_idx] & (1 << bit_idx)) == 0:
                count += 1
    return count


def test_null_count(n):
    """Test null_count for given size."""
    # Create input bitmap
    bitmap_size = (n + 7) // 8
    valid_np = np.random.randint(0, 256, bitmap_size, dtype=np.uint8)
    count_np = np.zeros(1, dtype=np.int64)

    # Copy to device
    valid_d = cuda.to_device(valid_np)
    count_d = cuda.to_device(count_np)

    # Run kernel
    module.run(valid_d, count_d, n)

    # Copy results back
    count_result = count_d.copy_to_host()[0]

    # Compute reference
    ref = cpu_reference(valid_np, n)

    # Check results
    assert count_result == ref, f"Count mismatch for n={n}: got {count_result}, expected {ref}"

    print(f"Test passed for n={n}: {count_result} nulls")


if __name__ == "__main__":
    for n in [0, 1, 1000, 1000003]:
        test_null_count(n)
    print("All tests passed!")
