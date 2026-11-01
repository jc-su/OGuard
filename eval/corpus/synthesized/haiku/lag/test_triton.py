import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/lag/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_lag(n, k):
    # Create test data
    values = np.random.rand(n).astype(np.float64)

    # Create partition info: first[i] is the first row of row i's partition
    # Build partitions of varying sizes
    first = np.zeros(n, dtype=np.int32)
    partition_start = 0
    i = 0
    while i < n:
        partition_size = np.random.randint(1, 20)
        partition_end = min(partition_start + partition_size, n)
        for j in range(partition_start, partition_end):
            first[j] = partition_start
        partition_start = partition_end
        i = partition_end

    # Compute CPU reference
    out_ref = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i - k >= first[i]:
            out_ref[i] = values[i - k]
        else:
            out_ref[i] = np.nan

    # GPU computation
    device = torch.device('cuda:0')
    values_gpu = torch.from_numpy(values).to(device)
    first_gpu = torch.from_numpy(first).to(device)
    out_gpu = torch.zeros(n, dtype=torch.float64, device=device)

    run(values_gpu, first_gpu, out_gpu, n, k)

    out_result = out_gpu.cpu().numpy()

    # Check results - handle NaN comparison
    mask_ref_nan = np.isnan(out_ref)
    mask_result_nan = np.isnan(out_result)

    if not (np.array_equal(mask_ref_nan, mask_result_nan) and
            np.allclose(out_result[~mask_result_nan], out_ref[~mask_ref_nan], rtol=1e-10)):
        print(f"FAILED for n={n}, k={k}")
        print(f"NaN mask match: {np.array_equal(mask_ref_nan, mask_result_nan)}")
        non_nan_mask = ~mask_result_nan & ~mask_ref_nan
        if non_nan_mask.any():
            print(f"Max diff (non-NaN): {np.abs(out_result[non_nan_mask] - out_ref[non_nan_mask]).max()}")
        return False
    print(f"PASSED for n={n}, k={k}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n in [0, 1, 1000, 1000003]:
        for k in [1, 5, 16]:
            all_pass = all_pass and test_lag(n, k)

    if all_pass:
        print("\nAll lag tests passed!")
    else:
        print("\nSome lag tests failed!")
        sys.exit(1)
