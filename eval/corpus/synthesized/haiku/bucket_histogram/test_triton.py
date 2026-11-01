import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/bucket_histogram/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_bucket_histogram(values, bounds):
    """Reference CPU implementation."""
    b = len(bounds)
    counts = np.zeros(b + 1, dtype=np.int32)

    for val in values:
        # Count how many bounds are <= val
        bucket = 0
        for bound in bounds:
            if bound <= val:
                bucket += 1
            else:
                break
        counts[bucket] += 1

    return counts


def test_bucket_histogram(n, b):
    """Test bucket_histogram kernel."""
    # Create bounds
    if b == 0:
        bounds = np.array([], dtype=np.float64)
    else:
        bounds = np.array([float(i) for i in range(b)], dtype=np.float64)

    # Create values
    if n == 0:
        values = np.array([], dtype=np.float64)
    else:
        values = np.array([float(i % (b + 2)) - 0.5 for i in range(n)], dtype=np.float64)

    # Compute reference on CPU
    expected = numpy_bucket_histogram(values, bounds)

    # Compute on GPU
    values_gpu = torch.from_numpy(values).cuda()
    bounds_gpu = torch.from_numpy(bounds).cuda()
    counts_gpu = torch.zeros(b + 1, dtype=torch.int32, device='cuda')

    run(values_gpu, bounds_gpu, counts_gpu, n, b)

    # Compare results
    counts_cpu = counts_gpu.cpu().numpy()

    if np.array_equal(counts_cpu, expected):
        print(f"  n={n}, b={b}: PASS")
        return True
    else:
        print(f"  n={n}, b={b}: FAIL")
        print(f"    Expected: {expected}")
        print(f"    Got:      {counts_cpu}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("Testing bucket_histogram kernel:")
    all_passed = True

    for n in [0, 1, 1000, 1000003]:
        b = max(1, n // 100)
        if not test_bucket_histogram(n, b):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
