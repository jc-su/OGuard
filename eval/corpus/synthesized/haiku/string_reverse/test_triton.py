import torch
import numpy as np
import sys
import importlib.util

# Load triton.py by path to avoid shadowing the triton package
spec = importlib.util.spec_from_file_location("triton_kernel", "/opgate/eval/corpus/synthesized/haiku/string_reverse/triton.py")
triton_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_module)
run = triton_module.run


def numpy_string_reverse(offsets, chars):
    """Reference CPU implementation."""
    total_len = offsets[-1]
    out = np.zeros(total_len, dtype=np.uint8)
    n = len(offsets) - 1

    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        # Reverse the string
        out[start:end] = chars[start:end][::-1]

    return out


def test_string_reverse(n_strings):
    """Test string_reverse kernel."""
    # Create test data with strings
    if n_strings == 0:
        offsets = np.array([0], dtype=np.int32)
        chars = np.array([], dtype=np.uint8)
    else:
        # Create n_strings strings with varying lengths
        string_lengths = np.array([10 * (i % 10 + 1) for i in range(n_strings)], dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(string_lengths)]).astype(np.int32)
        total_len = offsets[-1]
        chars = np.arange(total_len, dtype=np.int32) % 256
        chars = chars.astype(np.uint8)

    # Compute reference on CPU
    expected = numpy_string_reverse(offsets, chars)

    # Compute on GPU
    offsets_gpu = torch.from_numpy(offsets).cuda()
    chars_gpu = torch.from_numpy(chars).cuda()
    out_gpu = torch.zeros_like(chars_gpu)

    run(offsets_gpu, chars_gpu, out_gpu, n_strings)

    # Compare results
    out_cpu = out_gpu.cpu().numpy()

    if np.array_equal(out_cpu, expected):
        print(f"  n_strings={n_strings}: PASS")
        return True
    else:
        print(f"  n_strings={n_strings}: FAIL")
        return False


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print("Testing string_reverse kernel:")
    all_passed = True

    for n in [0, 1, 100, 10000]:
        if not test_string_reverse(n):
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)
