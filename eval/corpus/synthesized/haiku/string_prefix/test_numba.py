import numpy as np
from numba import cuda
import importlib.util

spec = importlib.util.spec_from_file_location("string_prefix_kernel", "$HOME/Dev/opgate-heldout4/haiku/string_prefix/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(offsets, chars, n, k):
    """Compute reference output on CPU."""
    out_ref = np.zeros((n, k), dtype=np.uint8)
    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        length = min(end - start, k)
        for j in range(length):
            out_ref[i, j] = chars[start + j]
    return out_ref


def test_string_prefix(n, k):
    """Test string_prefix for a given size."""
    # Create input data with variable-length strings
    offsets_np = np.zeros(n + 1, dtype=np.int32)
    total_chars = 0
    for i in range(n):
        str_len = (i * 7 + 3) % 50  # Variable length strings
        offsets_np[i + 1] = offsets_np[i] + str_len
        total_chars += str_len

    chars_np = np.random.randint(32, 127, total_chars, dtype=np.uint8)
    out_np = np.zeros(n * k, dtype=np.uint8)

    # Copy to device
    offsets_d = cuda.to_device(offsets_np)
    chars_d = cuda.to_device(chars_np)
    out_d = cuda.to_device(out_np)

    # Run kernel
    module.run(offsets_d, chars_d, out_d, n, k)

    # Copy results back
    out_result = out_d.copy_to_host()

    # Compute reference
    ref = cpu_reference(offsets_np, chars_np, n, k)

    # Check results
    out_result_reshaped = out_result.reshape((n, k))
    assert np.array_equal(out_result_reshaped, ref), f"Values mismatch for n={n}, k={k}"

    print(f"Test passed for n={n}, k={k}")


if __name__ == "__main__":
    for n in [0, 1, 1000, 1000003]:
        for k in [1, 2, 32, 64]:
            test_string_prefix(n, k)
    print("All tests passed!")
