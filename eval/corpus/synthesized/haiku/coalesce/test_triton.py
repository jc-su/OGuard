import torch
import numpy as np
import sys
import os

# Import triton before adding local path to avoid shadowing
import triton
import triton.language as tl

sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/coalesce')
# Now we can safely do relative import using importlib
import importlib.util
spec = importlib.util.spec_from_file_location("kernel_module", "/opgate/eval/corpus/synthesized/haiku/coalesce/triton.py")
kernel_module = importlib.util.module_from_spec(spec)
sys.modules["kernel_module"] = kernel_module
spec.loader.exec_module(kernel_module)
run = kernel_module.run

def test_coalesce(n):
    # Create test data
    a = np.random.rand(n).astype(np.float64)
    b = np.random.rand(n).astype(np.float64)

    # Create validity bitmap
    valid_bytes = (n + 7) // 8
    valid = np.random.randint(0, 256, valid_bytes, dtype=np.uint8)

    # Compute CPU reference
    out_ref = np.zeros(n, dtype=np.float64)
    for i in range(n):
        byte_idx = i // 8
        bit_idx = i % 8
        is_valid = (valid[byte_idx] >> bit_idx) & 1
        out_ref[i] = a[i] if is_valid else b[i]

    # GPU computation
    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES=1 makes it appear as cuda:0
    a_gpu = torch.from_numpy(a).to(device)
    valid_gpu = torch.from_numpy(valid).to(device)
    b_gpu = torch.from_numpy(b).to(device)
    out_gpu = torch.zeros(n, dtype=torch.float64, device=device)

    run(a_gpu, valid_gpu, b_gpu, out_gpu, n)

    out_result = out_gpu.cpu().numpy()

    # Check results
    if not np.allclose(out_result, out_ref, rtol=1e-10):
        print(f"FAILED for n={n}")
        print(f"Max diff: {np.abs(out_result - out_ref).max()}")
        return False
    print(f"PASSED for n={n}")
    return True

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    all_pass = True
    for n in [0, 1, 1000, 1000003]:
        all_pass = all_pass and test_coalesce(n)

    if all_pass:
        print("\nAll coalesce tests passed!")
    else:
        print("\nSome coalesce tests failed!")
        sys.exit(1)
