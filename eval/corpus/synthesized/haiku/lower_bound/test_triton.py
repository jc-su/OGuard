import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/lower_bound/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_lower_bound(keys, sorted_arr):
    """Reference CPU implementation using searchsorted."""
    return np.searchsorted(sorted_arr, keys, side='left').astype(np.int32)


def test_lower_bound(n, s):
    """Test lower_bound kernel."""
    # Create test data
    if s == 0:
        sorted_arr = np.array([], dtype=np.int64)
    else:
        # Create a sorted array
        sorted_arr = np.sort(np.arange(s, dtype=np.int64) % 100)

    if n == 0:
        keys = np.array([], dtype=np.int64)
    else:
        # Create keys with some in range and some outside
        keys = np.array([i % (max(1, s)) for i in range(n)], dtype=np.int64)

    # Compute reference on CPU
    expected = numpy_lower_bound(keys, sorted_arr)

    # Compute on GPU
    keys_gpu = torch.from_numpy(keys).cuda()
    sorted_gpu = torch.from_numpy(sorted_arr).cuda()
    out_gpu = torch.zeros(n, dtype=torch.int32, device='cuda')

    run(keys_gpu, sorted_gpu, out_gpu, n, s)

    # Compare results
    out_cpu = out_gpu.cpu().numpy()

    if np.array_equal(out_cpu, expected):
        print(f"  n={n}, s={s}: PASS")
        return True
    else:
        print(f"  n={n}, s={s}: FAIL")
        print(f"    Expected: {expected[:min(10, n)]}")
        print(f"    Got:      {out_cpu[:min(10, n)]}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print("Testing lower_bound kernel:")
    all_passed = True

    for n in [0, 1, 1000, 1000003]:
        s = max(1, n // 10)
        if not test_lower_bound(n, s):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
