import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/gather_nullable/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_gather_nullable(m, n):
    # Create test data
    col = np.random.rand(n).astype(np.float64)
    # Create indices: mix of valid indices and -1
    idx = np.random.randint(-1, n, m, dtype=np.int32)

    # Compute CPU reference
    out_ref = np.zeros(m, dtype=np.float64)
    for i in range(m):
        if idx[i] == -1:
            out_ref[i] = np.nan
        else:
            out_ref[i] = col[idx[i]]

    # GPU computation
    device = torch.device('cuda:0')
    col_gpu = torch.from_numpy(col).to(device)
    idx_gpu = torch.from_numpy(idx).to(device)
    out_gpu = torch.zeros(m, dtype=torch.float64, device=device)

    run(col_gpu, idx_gpu, out_gpu, m, n)

    out_result = out_gpu.cpu().numpy()

    # Check results - need to handle NaN comparison
    mask_ref_nan = np.isnan(out_ref)
    mask_result_nan = np.isnan(out_result)

    if not (np.array_equal(mask_ref_nan, mask_result_nan) and
            np.allclose(out_result[~mask_result_nan], out_ref[~mask_ref_nan], rtol=1e-10)):
        print(f"FAILED for m={m}, n={n}")
        print(f"NaN mask match: {np.array_equal(mask_ref_nan, mask_result_nan)}")
        non_nan_mask = ~mask_result_nan & ~mask_ref_nan
        if non_nan_mask.any():
            print(f"Max diff (non-NaN): {np.abs(out_result[non_nan_mask] - out_ref[non_nan_mask]).max()}")
        return False
    print(f"PASSED for m={m}, n={n}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for m in [0, 1, 1000, 1000003]:
        n = max(1, m)  # Need at least one element in col
        all_pass = all_pass and test_gather_nullable(m, n)

    if all_pass:
        print("\nAll gather_nullable tests passed!")
    else:
        print("\nSome gather_nullable tests failed!")
        sys.exit(1)
