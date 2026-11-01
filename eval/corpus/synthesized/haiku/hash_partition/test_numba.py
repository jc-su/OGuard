import numpy as np
from numba import cuda
import importlib.util

spec = importlib.util.spec_from_file_location("hash_partition_kernel", "$HOME/Dev/opgate-heldout4/haiku/hash_partition/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def hash_func(x):
    """Hash function matching the kernel."""
    # Use numpy for safe integer arithmetic
    x_val = np.uint64(x)
    h = (x_val * np.uint64(0x9E3779B97F4A7C15)) >> 32
    return int(h)


def cpu_reference(keys, n, p):
    """Compute reference output on CPU."""
    part_ref = np.zeros(n, dtype=np.int32)
    counts_ref = np.zeros(p, dtype=np.int32)
    for i in range(n):
        h = hash_func(keys[i])
        partition = h % p
        part_ref[i] = partition
        counts_ref[partition] += 1
    return part_ref, counts_ref


def test_hash_partition(n, p):
    """Test hash_partition for given parameters."""
    # Create input data
    keys_np = np.random.randint(-2**63, 2**63-1, n, dtype=np.int64)
    part_np = np.zeros(n, dtype=np.int32)
    counts_np = np.zeros(p, dtype=np.int32)

    # Copy to device
    keys_d = cuda.to_device(keys_np)
    part_d = cuda.to_device(part_np)
    counts_d = cuda.to_device(counts_np)

    # Run kernel
    module.run(keys_d, part_d, counts_d, n, p)

    # Copy results back
    part_result = part_d.copy_to_host()
    counts_result = counts_d.copy_to_host()

    # Compute reference
    part_ref, counts_ref = cpu_reference(keys_np, n, p)

    # Check results
    assert np.array_equal(part_result, part_ref), f"Part mismatch for n={n}, p={p}"
    assert np.array_equal(counts_result, counts_ref), f"Counts mismatch for n={n}, p={p}"

    print(f"Test passed for n={n}, p={p}")


if __name__ == "__main__":
    test_cases = [(0, 1), (0, 16), (1, 1), (1, 16), (1000, 16), (1000003, 256)]
    for n, p in test_cases:
        test_hash_partition(n, p)
    print("All tests passed!")
