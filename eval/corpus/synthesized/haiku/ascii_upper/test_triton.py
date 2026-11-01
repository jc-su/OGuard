import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/ascii_upper/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_ascii_upper(n):
    # Create test data: n strings with mixed case ASCII characters
    chars_list = []
    offsets = [0]

    for i in range(n):
        # Random string of 0-100 characters with mix of cases
        str_len = np.random.randint(0, 100)
        if str_len > 0:
            # Mix of lowercase, uppercase, and non-letter ASCII
            chars = np.random.randint(32, 127, str_len, dtype=np.uint8)
        else:
            chars = np.array([], dtype=np.uint8)
        chars_list.append(chars)
        offsets.append(offsets[-1] + str_len)

    chars = np.concatenate(chars_list) if chars_list else np.array([], dtype=np.uint8)
    offsets = np.array(offsets, dtype=np.int32)

    # Compute CPU reference
    out_ref = chars.copy()
    for i in range(len(out_ref)):
        if 97 <= out_ref[i] <= 122:  # lowercase a-z
            out_ref[i] = out_ref[i] - 32  # convert to uppercase

    # GPU computation
    device = torch.device('cuda:0')
    offsets_gpu = torch.from_numpy(offsets).to(device)
    chars_gpu = torch.from_numpy(chars).to(device)
    out_gpu = torch.zeros_like(chars_gpu)

    run(offsets_gpu, chars_gpu, out_gpu, n)

    out_result = out_gpu.cpu().numpy()

    # Check results
    if not np.array_equal(out_result, out_ref):
        print(f"FAILED for n={n}")
        diff_idx = np.where(out_result != out_ref)[0]
        if len(diff_idx) > 0:
            idx = diff_idx[0]
            print(f"First mismatch at index {idx}: got {out_result[idx]}, expected {out_ref[idx]}")
        return False
    print(f"PASSED for n={n}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n in [0, 1, 1000, 1000003]:
        all_pass = all_pass and test_ascii_upper(n)

    if all_pass:
        print("\nAll ascii_upper tests passed!")
    else:
        print("\nSome ascii_upper tests failed!")
        sys.exit(1)
