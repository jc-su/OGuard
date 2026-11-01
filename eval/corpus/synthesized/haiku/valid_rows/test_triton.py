import torch
import numpy as np
import importlib.util
import sys


def load_module(path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location("triton_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_valid_rows(valid, n):
    """CPU reference implementation."""
    rows = []
    for i in range(n):
        byte_idx = i // 8
        bit_idx = i % 8
        if byte_idx < len(valid):
            is_valid = (valid[byte_idx] >> bit_idx) & 1
            if is_valid:
                rows.append(i)
    return rows


def test_valid_rows():
    """Test valid_rows kernel on various input sizes."""
    # Load the triton kernel
    triton_module = load_module("/scratch/heldout/haiku/valid_rows/triton.py")

    test_sizes = [0, 1, 1000, 1000003]

    for n in test_sizes:
        print(f"Testing valid_rows with n={n}")

        # Create random validity bitmap
        valid_bytes = np.random.randint(0, 256, size=(n + 7) // 8, dtype=np.uint8)
        valid_device = torch.from_numpy(valid_bytes).cuda()

        # Allocate output arrays
        rows_device = torch.zeros(n, dtype=torch.int32, device='cuda')
        count_device = torch.zeros(1, dtype=torch.int32, device='cuda')

        # Run the kernel
        triton_module.run(valid_device, rows_device, count_device, n)

        # Get results
        count_result = count_device.item()
        rows_result = rows_device[:count_result].cpu().numpy()

        # CPU reference
        expected_rows = cpu_valid_rows(valid_bytes, n)

        # Check results
        assert len(rows_result) == len(expected_rows), f"Count mismatch: {len(rows_result)} vs {len(expected_rows)}"
        assert count_result == len(expected_rows), f"Count value mismatch: {count_result} vs {len(expected_rows)}"
        assert np.array_equal(np.sort(rows_result), np.sort(expected_rows)), f"Rows mismatch"

        print(f"  PASS: count={count_result}, rows match")


if __name__ == "__main__":
    test_valid_rows()
    print("All tests passed!")
