import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/group_max/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_group_max(group, value, capacity):
    """Reference CPU implementation."""
    best = np.full(capacity, -(2 ** 63), dtype=np.int64)
    for i in range(len(group)):
        g = group[i]
        best[g] = max(best[g], value[i])
    return best


def test_group_max(n, capacity):
    """Test group_max kernel."""
    if n == 0:
        group = np.array([], dtype=np.int32)
        value = np.array([], dtype=np.int64)
    else:
        group = np.array([i % capacity for i in range(n)], dtype=np.int32)
        value = np.array([i % 100 for i in range(n)], dtype=np.int64)

    # Compute reference on CPU
    expected = numpy_group_max(group, value, capacity)

    # Compute on GPU
    group_gpu = torch.from_numpy(group).cuda()
    value_gpu = torch.from_numpy(value).cuda()
    best_gpu = torch.full((capacity,), -(2 ** 63), dtype=torch.int64, device='cuda')

    run(group_gpu, value_gpu, best_gpu, n, capacity)

    # Compare results
    best_cpu = best_gpu.cpu().numpy()

    if np.array_equal(best_cpu, expected):
        print(f"  n={n}, capacity={capacity}: PASS")
        return True
    else:
        print(f"  n={n}, capacity={capacity}: FAIL")
        print(f"    Expected: {expected[:min(10, capacity)]}")
        print(f"    Got:      {best_cpu[:min(10, capacity)]}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("Testing group_max kernel:")
    all_passed = True

    for n in [0, 1, 1000, 1000003]:
        capacity = max(1, n // 10)
        if not test_group_max(n, capacity):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
