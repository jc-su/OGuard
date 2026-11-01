import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/bucketize/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_bucketize(n, b):
    # Create test data
    values = np.random.rand(n).astype(np.float64) * 100  # Values 0-100
    bounds = np.sort(np.random.rand(b).astype(np.float64) * 100)  # Sorted bounds 0-100

    # Compute CPU reference
    out_ref = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # Count how many bounds are <= values[i]
        count = 0
        for j in range(b):
            if bounds[j] <= values[i]:
                count += 1
        out_ref[i] = count

    # GPU computation
    device = torch.device('cuda:0')
    values_gpu = torch.from_numpy(values).to(device)
    bounds_gpu = torch.from_numpy(bounds).to(device)
    out_gpu = torch.zeros(n, dtype=torch.int32, device=device)

    run(values_gpu, bounds_gpu, out_gpu, n, b)

    out_result = out_gpu.cpu().numpy()

    # Check results
    if not np.array_equal(out_result, out_ref):
        print(f"FAILED for n={n}, b={b}")
        diff_idx = np.where(out_result != out_ref)[0]
        if len(diff_idx) > 0:
            idx = diff_idx[0]
            print(f"First mismatch at index {idx}: got {out_result[idx]}, expected {out_ref[idx]}")
            print(f"  Value: {values[idx]}, Bounds: {bounds}")
        return False
    print(f"PASSED for n={n}, b={b}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n in [0, 1, 1000, 100000]:
        for b in [10, 100]:
            all_pass = all_pass and test_bucketize(n, b)

    if all_pass:
        print("\nAll bucketize tests passed!")
    else:
        print("\nSome bucketize tests failed!")
        sys.exit(1)
