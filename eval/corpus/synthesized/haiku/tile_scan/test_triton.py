import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/tile_scan/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_tile_scan(n):
    # Create test data
    values = np.random.randint(1, 10, n, dtype=np.int32)

    # Compute CPU reference
    TILE_SIZE = 1024
    out_ref = np.zeros(n, dtype=np.int32)
    totals_ref = np.zeros((n + TILE_SIZE - 1) // TILE_SIZE, dtype=np.int32)

    for i in range(n):
        tile_id = i // TILE_SIZE
        tile_start = tile_id * TILE_SIZE

        # Compute prefix sum within tile
        prefix_sum = 0
        for j in range(tile_start, i + 1):
            prefix_sum += values[j]
        out_ref[i] = prefix_sum

        # Compute tile total (only once per tile)
        if i == min(tile_start + TILE_SIZE - 1, n - 1):  # Last row of tile
            total = 0
            for j in range(tile_start, min(tile_start + TILE_SIZE, n)):
                total += values[j]
            totals_ref[tile_id] = total

    # GPU computation
    device = torch.device('cuda:0')
    values_gpu = torch.from_numpy(values).to(device)
    out_gpu = torch.zeros(n, dtype=torch.int32, device=device)
    totals_gpu = torch.zeros((n + TILE_SIZE - 1) // TILE_SIZE, dtype=torch.int32, device=device)

    run(values_gpu, out_gpu, totals_gpu, n)

    out_result = out_gpu.cpu().numpy()
    totals_result = totals_gpu.cpu().numpy()

    # Check results
    out_match = np.array_equal(out_result, out_ref)
    totals_match = np.array_equal(totals_result, totals_ref)

    if not (out_match and totals_match):
        print(f"FAILED for n={n}")
        if not out_match:
            diff_idx = np.where(out_result != out_ref)[0]
            if len(diff_idx) > 0:
                idx = diff_idx[0]
                print(f"  Out mismatch at index {idx}: got {out_result[idx]}, expected {out_ref[idx]}")
        if not totals_match:
            diff_idx = np.where(totals_result != totals_ref)[0]
            if len(diff_idx) > 0:
                idx = diff_idx[0]
                print(f"  Totals mismatch at index {idx}: got {totals_result[idx]}, expected {totals_ref[idx]}")
        return False
    print(f"PASSED for n={n}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n in [0, 1, 1000, 100000]:
        all_pass = all_pass and test_tile_scan(n)

    if all_pass:
        print("\nAll tile_scan tests passed!")
    else:
        print("\nSome tile_scan tests failed!")
        sys.exit(1)
