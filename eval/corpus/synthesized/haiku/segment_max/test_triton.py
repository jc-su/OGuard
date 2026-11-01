import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/segment_max/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_segment_max(offsets, values):
    """Reference CPU implementation."""
    g = len(offsets) - 1
    maxes = np.full(g, -(2 ** 63), dtype=np.int64)
    for j in range(g):
        seg_start = offsets[j]
        seg_end = offsets[j + 1]
        if seg_start < seg_end:
            maxes[j] = np.max(values[seg_start:seg_end])
    return maxes


def test_segment_max(n, g):
    """Test segment_max kernel."""
    # Create test data
    if n == 0:
        values = np.array([], dtype=np.int64)
        offsets = np.zeros(g + 1, dtype=np.int32)
    else:
        values = np.arange(n, dtype=np.int64) % 100
        # Create random segment boundaries
        np.random.seed(42)
        if g == 0:
            offsets = np.array([0], dtype=np.int32)
        else:
            # Create g segments
            boundaries = sorted(np.random.choice(n, g - 1, replace=False))
            offsets = np.concatenate([[0], boundaries, [n]]).astype(np.int32)
            g = len(offsets) - 1

    # Compute reference on CPU
    expected = numpy_segment_max(offsets, values)

    # Compute on GPU
    offsets_gpu = torch.from_numpy(offsets).cuda()
    values_gpu = torch.from_numpy(values).cuda()
    maxes_gpu = torch.zeros(g, dtype=torch.int64, device='cuda')

    run(offsets_gpu, values_gpu, maxes_gpu, g)

    # Compare results
    maxes_cpu = maxes_gpu.cpu().numpy()

    if np.array_equal(maxes_cpu, expected):
        print(f"  n={n}, g={g}: PASS")
        return True
    else:
        print(f"  n={n}, g={g}: FAIL")
        print(f"    Expected: {expected[:min(10, g)]}")
        print(f"    Got:      {maxes_cpu[:min(10, g)]}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print("Testing segment_max kernel:")
    all_passed = True

    for n in [0, 1, 1000, 1000003]:
        g = max(1, n // 100)  # Create multiple segments for larger n
        if not test_segment_max(n, g):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
