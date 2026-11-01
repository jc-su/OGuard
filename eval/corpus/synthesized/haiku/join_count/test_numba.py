import numpy as np
from numba import cuda
import importlib.util

spec = importlib.util.spec_from_file_location("join_count_kernel", "$HOME/Dev/opgate-heldout4/haiku/join_count/numba.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cpu_reference(build, probe, b, p):
    """Compute reference output on CPU."""
    counts_ref = np.zeros(p, dtype=np.int32)
    for i in range(p):
        counts_ref[i] = np.sum(build[:b] == probe[i])
    return counts_ref


def test_join_count(b, p):
    """Test join_count for given sizes."""
    # Create input data
    if b == 0:
        build_np = np.array([], dtype=np.int64)
        probe_np = np.random.randint(0, 100, p, dtype=np.int64)
    else:
        build_np = np.sort(np.random.randint(0, 100, b, dtype=np.int64))
        probe_np = np.random.randint(0, 100, p, dtype=np.int64)

    counts_np = np.zeros(p, dtype=np.int32)

    # Copy to device
    build_d = cuda.to_device(build_np)
    probe_d = cuda.to_device(probe_np)
    counts_d = cuda.to_device(counts_np)

    # Run kernel
    module.run(build_d, probe_d, counts_d, b, p)

    # Copy results back
    counts_result = counts_d.copy_to_host()

    # Compute reference
    ref = cpu_reference(build_np, probe_np, b, p)

    # Check results
    assert np.array_equal(counts_result, ref), f"Values mismatch for b={b}, p={p}"

    print(f"Test passed for b={b}, p={p}")


if __name__ == "__main__":
    for b in [0, 1, 1000, 1000003]:
        for p in [1, 100, 1000, 10000]:
            test_join_count(b, p)
    print("All tests passed!")
