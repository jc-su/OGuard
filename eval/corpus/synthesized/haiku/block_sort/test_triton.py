import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/block_sort/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_block_sort(n):
    # Create test data
    keys = np.random.randint(0, 1000000, n, dtype=np.int32)

    # Compute CPU reference
    out_ref = keys.copy()
    TILE_SIZE = 1024
    for tile_id in range((n + TILE_SIZE - 1) // TILE_SIZE):
        tile_start = tile_id * TILE_SIZE
        tile_end = min(tile_start + TILE_SIZE, n)
        # Sort this tile
        out_ref[tile_start:tile_end] = np.sort(out_ref[tile_start:tile_end])

    # GPU computation
    device = torch.device('cuda:0')
    keys_gpu = torch.from_numpy(keys).to(device)
    out_gpu = torch.zeros(n, dtype=torch.int32, device=device)

    run(keys_gpu, out_gpu, n)

    out_result = out_gpu.cpu().numpy()

    # Check results
    if not np.array_equal(out_result, out_ref):
        print(f"FAILED for n={n}")
        diff_idx = np.where(out_result != out_ref)[0]
        if len(diff_idx) > 0:
            idx = diff_idx[0]
            print(f"First mismatch at index {idx}: got {out_result[idx]}, expected {out_ref[idx]}")
            # Find which tile
            tile_id = idx // 1024
            print(f"  In tile {tile_id} (range {tile_id*1024}-{min((tile_id+1)*1024, n)})")
        return False
    print(f"PASSED for n={n}")
    return True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n in [0, 1, 1000, 10000]:
        all_pass = all_pass and test_block_sort(n)

    if all_pass:
        print("\nAll block_sort tests passed!")
    else:
        print("\nSome block_sort tests failed!")
        sys.exit(1)
