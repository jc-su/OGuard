import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/masked_sum/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_masked_sum(values, mask):
    """Reference CPU implementation."""
    result = np.sum(values[mask != 0])
    return result


def test_masked_sum(n):
    """Test masked_sum kernel."""
    if n == 0:
        values = np.array([], dtype=np.float64)
        mask = np.array([], dtype=np.int8)
    else:
        values = np.array([float(i) / 10.0 for i in range(n)], dtype=np.float64)
        mask = np.array([i % 2 for i in range(n)], dtype=np.int8)  # Alternate 0 and 1

    # Compute reference on CPU
    expected = numpy_masked_sum(values, mask)

    # Compute on GPU
    values_gpu = torch.from_numpy(values).cuda()
    mask_gpu = torch.from_numpy(mask).cuda()
    result_gpu = torch.zeros(1, dtype=torch.float64, device='cuda')

    run(values_gpu, mask_gpu, result_gpu, n)

    # Compare results
    result_cpu = result_gpu.cpu().numpy()[0]

    # Use approximate equality for floating point
    if np.isclose(result_cpu, expected, rtol=1e-9):
        print(f"  n={n}: PASS")
        return True
    else:
        print(f"  n={n}: FAIL")
        print(f"    Expected: {expected}")
        print(f"    Got:      {result_cpu}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("Testing masked_sum kernel:")
    all_passed = True

    for n in [0, 1, 1000, 1000003]:
        if not test_masked_sum(n):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
