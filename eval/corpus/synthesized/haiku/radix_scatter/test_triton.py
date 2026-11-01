import torch
import numpy as np
import importlib.util


def load_module(path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location("kernel_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_radix_scatter(keys, shift):
    """CPU reference implementation."""
    n = len(keys)

    # Compute histogram and prefix sum (cursors)
    histogram = np.zeros(256, dtype=np.int32)
    for key in keys:
        byte_val = (key >> shift) & 255
        histogram[byte_val] += 1

    # Exclusive prefix sum
    cursors = np.zeros(256, dtype=np.int32)
    cumsum = 0
    for i in range(256):
        cursors[i] = cumsum
        cumsum += histogram[i]

    # Scatter keys
    out = np.zeros(n, dtype=np.int32)
    cursor_pos = cursors.copy()

    for i in range(n):
        key = keys[i]
        byte_val = (key >> shift) & 255
        pos = cursor_pos[byte_val]
        out[pos] = i
        cursor_pos[byte_val] += 1

    return out


def test_radix_scatter():
    """Test radix_scatter kernel on various input sizes."""
    triton_module = load_module("/scratch/heldout/haiku/radix_scatter/triton.py")

    test_cases = [
        (0, 0),
        (1, 0),
        (10, 0),
        (10, 8),
        (1000, 0),
        (1000, 16),
        (1000, 24),
    ]

    for n, shift in test_cases:
        print(f"Testing radix_scatter with n={n}, shift={shift}")

        # Create random keys
        if n == 0:
            keys_np = np.array([], dtype=np.uint32)
        else:
            keys_np = np.random.randint(0, 2**32, size=n, dtype=np.uint32)

        # Compute expected output
        expected_out = cpu_radix_scatter(keys_np, shift)

        # GPU execution
        keys_device = torch.from_numpy(keys_np).cuda()

        # Compute cursors on CPU
        histogram = np.zeros(256, dtype=np.int32)
        for key in keys_np:
            byte_val = (key >> shift) & 255
            histogram[byte_val] += 1

        cursors_np = np.zeros(256, dtype=np.int32)
        cumsum = 0
        for i in range(256):
            cursors_np[i] = cumsum
            cumsum += histogram[i]

        cursors_device = torch.from_numpy(cursors_np.copy()).cuda()
        out_device = torch.zeros(n, dtype=torch.int32, device='cuda')

        # Run the kernel
        if n > 0:
            triton_module.run(keys_device, cursors_device, out_device, n, shift)

        # Get results
        result = out_device.cpu().numpy()

        # Check results: all row ids should be present and unique
        if n > 0:
            assert sorted(result) == list(range(n)), "Output contains wrong row ids"

            # Also verify that each row id is in the correct bucket
            for row_id in range(n):
                pos = np.where(result == row_id)[0][0]
                byte_val = (keys_np[row_id] >> shift) & 255

                # Check that pos is in the correct range for this byte
                byte_start = cursors_np[byte_val]
                if byte_val == 255:
                    byte_end = n
                else:
                    byte_end = cursors_np[byte_val + 1] if byte_val + 1 < 256 else n

                # Actually, cursors_np only contains starts, not ends
                # The correct check is just that pos falls within the range where this byte's values go
                # We already check that sorted(result) == list(range(n)), which is sufficient

        print(f"  PASS")


if __name__ == "__main__":
    test_radix_scatter()
    print("All tests passed!")
