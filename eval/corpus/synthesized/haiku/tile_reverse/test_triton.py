import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/tile_reverse/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_tile_reverse(values):
    """Reference CPU implementation."""
    n = len(values)
    out = np.zeros(n, dtype=np.int32)

    tile_size = 1024
    tile_idx = 0

    while tile_idx * tile_size < n:
        tile_start = tile_idx * tile_size
        tile_end = min((tile_idx + 1) * tile_size, n)
        tile_len = tile_end - tile_start

        # Reverse this tile
        out[tile_start:tile_end] = values[tile_start:tile_end][::-1]

        tile_idx += 1

    return out


def test_tile_reverse(n):
    """Test tile_reverse kernel."""
    if n == 0:
        values = np.array([], dtype=np.int32)
    else:
        values = np.array([i % 100 for i in range(n)], dtype=np.int32)

    # Compute reference on CPU
    expected = numpy_tile_reverse(values)

    # Compute on GPU
    values_gpu = torch.from_numpy(values).cuda()
    out_gpu = torch.zeros(n, dtype=torch.int32, device='cuda')

    run(values_gpu, out_gpu, n)

    # Compare results
    out_cpu = out_gpu.cpu().numpy()

    if np.array_equal(out_cpu, expected):
        print(f"  n={n}: PASS")
        return True
    else:
        print(f"  n={n}: FAIL")
        print(f"    Expected first 30: {expected[:min(30, n)]}")
        print(f"    Got first 30:      {out_cpu[:min(30, n)]}")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print("Testing tile_reverse kernel:")
    all_passed = True

    for n in [0, 1, 1000, 1000003]:
        if not test_tile_reverse(n):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
