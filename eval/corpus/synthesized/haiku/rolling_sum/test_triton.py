import torch
import numpy as np
import importlib.util


def load_module(path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location("kernel_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_rolling_sum(values, window):
    """CPU reference implementation."""
    n = len(values)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = np.sum(values[start:i+1])

    return out


def test_rolling_sum():
    """Test rolling_sum kernel on various input sizes."""
    triton_module = load_module("/scratch/heldout/haiku/rolling_sum/triton.py")

    test_cases = [
        (0, 1),
        (1, 1),
        (10, 1),
        (10, 3),
        (10, 10),
        (1000, 5),
        (1000, 256),
        (1000003, 7),
    ]

    for n, window in test_cases:
        print(f"Testing rolling_sum with n={n}, window={window}")

        # Create random values
        if n == 0:
            values_np = np.array([], dtype=np.float64)
        else:
            values_np = np.random.randn(n).astype(np.float64)

        values_device = torch.from_numpy(values_np).cuda()
        out_device = torch.zeros(n, dtype=torch.float64, device='cuda')

        # Run the kernel
        if n > 0:
            triton_module.run(values_device, out_device, n, window)

        # Get results
        result = out_device.cpu().numpy()

        # Expected results
        expected = cpu_rolling_sum(values_np, window)

        # Check results
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-15)
        print(f"  PASS")


if __name__ == "__main__":
    test_rolling_sum()
    print("All tests passed!")
