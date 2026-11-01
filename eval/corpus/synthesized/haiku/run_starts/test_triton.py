import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/run_starts/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_run_starts(values):
    """Reference CPU implementation."""
    n = len(values)
    flags = np.zeros(n, dtype=np.int8)
    if n > 0:
        flags[0] = 1
        for i in range(1, n):
            if values[i] != values[i - 1]:
                flags[i] = 1
    return flags


def test_run_starts(n):
    """Test run_starts kernel with size n."""
    # Create test data
    if n == 0:
        values = np.array([], dtype=np.int64)
    else:
        # Mix of repeated and changing values
        values = np.array([i % 3 for i in range(n)], dtype=np.int64)

    # Compute reference on CPU
    expected = numpy_run_starts(values)

    # Compute on GPU
    values_gpu = torch.from_numpy(values).cuda()
    flags_gpu = torch.zeros(n, dtype=torch.int8, device='cuda')

    run(values_gpu, flags_gpu, n)

    # Compare results
    flags_cpu = flags_gpu.cpu().numpy()

    if np.array_equal(flags_cpu, expected):
        print(f"  n={n}: PASS")
        return True
    else:
        print(f"  n={n}: FAIL")
        print(f"    Expected: {expected[:min(20, n)]}")
        print(f"    Got:      {flags_cpu[:min(20, n)]}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print("Testing run_starts kernel:")
    all_passed = True
    for n in [0, 1, 1000, 1000003]:
        if not test_run_starts(n):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
