import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/diff/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_diff(values):
    """Reference CPU implementation."""
    n = len(values)
    out = np.zeros(n, dtype=np.int64)
    if n > 0:
        out[0] = values[0]
        for i in range(1, n):
            out[i] = values[i] - values[i - 1]
    return out


def test_diff(n):
    """Test diff kernel."""
    if n == 0:
        values = np.array([], dtype=np.int64)
    else:
        values = np.array([i * 2 for i in range(n)], dtype=np.int64)

    # Compute reference on CPU
    expected = numpy_diff(values)

    # Compute on GPU
    values_gpu = torch.from_numpy(values).cuda()
    out_gpu = torch.zeros(n, dtype=torch.int64, device='cuda')

    run(values_gpu, out_gpu, n)

    # Compare results
    out_cpu = out_gpu.cpu().numpy()

    if np.array_equal(out_cpu, expected):
        print(f"  n={n}: PASS")
        return True
    else:
        print(f"  n={n}: FAIL")
        print(f"    Expected: {expected[:min(20, n)]}")
        print(f"    Got:      {out_cpu[:min(20, n)]}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("Testing diff kernel:")
    all_passed = True

    for n in [0, 1, 1000, 1000003]:
        if not test_diff(n):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
