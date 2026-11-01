import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/scatter_add/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_scatter_add(idx, values, n):
    """Reference CPU implementation."""
    out = np.zeros(n, dtype=np.float64)
    for i in range(len(idx)):
        out[idx[i]] += values[i]
    return out


def test_scatter_add(m, n):
    """Test scatter_add kernel."""
    if m == 0:
        idx = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.float64)
    else:
        idx = np.array([i % n for i in range(m)], dtype=np.int32)
        values = np.array([float(i) / 10.0 for i in range(m)], dtype=np.float64)

    # Compute reference on CPU
    expected = numpy_scatter_add(idx, values, n)

    # Compute on GPU
    idx_gpu = torch.from_numpy(idx).cuda()
    values_gpu = torch.from_numpy(values).cuda()
    out_gpu = torch.zeros(n, dtype=torch.float64, device='cuda')

    run(idx_gpu, values_gpu, out_gpu, m, n)

    # Compare results
    out_cpu = out_gpu.cpu().numpy()

    # Use approximate equality for floating point
    if np.allclose(out_cpu, expected, rtol=1e-9):
        print(f"  m={m}, n={n}: PASS")
        return True
    else:
        print(f"  m={m}, n={n}: FAIL")
        print(f"    Expected: {expected[:min(10, n)]}")
        print(f"    Got:      {out_cpu[:min(10, n)]}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print("Testing scatter_add kernel:")
    all_passed = True

    for m in [0, 1, 1000, 1000003]:
        n = max(1, m // 10)
        if not test_scatter_add(m, n):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
