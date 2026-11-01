import torch
import numpy as np
import importlib.util


def load_module(path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location("kernel_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_run_end_decode(run_values, run_ends, n):
    """CPU reference implementation."""
    out = np.zeros(n, dtype=np.float64)
    start = 0
    for i in range(len(run_values)):
        end = run_ends[i]
        out[start:end] = run_values[i]
        start = end
    return out


def test_run_end_decode():
    """Test run_end_decode kernel on various input sizes."""
    triton_module = load_module("/scratch/heldout/haiku/run_end_decode/triton.py")

    test_cases = [
        (0, []),  # Empty
        (1, [1]),  # Single row, single run
        (1000, None),  # Random runs
        (1000003, None),  # Large
    ]

    for n, run_ends_explicit in test_cases:
        print(f"Testing run_end_decode with n={n}")

        if n == 0:
            r = 0
            run_values_np = np.array([], dtype=np.float64)
            run_ends_np = np.array([], dtype=np.int32)
        else:
            # Create random run structure
            if run_ends_explicit:
                run_ends_np = np.array(run_ends_explicit, dtype=np.int32)
            else:
                # Generate random run ends
                num_runs = max(1, n // 1000 + 1)
                run_ends_list = sorted(set(np.random.choice(range(1, n+1), size=min(num_runs, n), replace=False)))
                if run_ends_list[-1] != n:
                    run_ends_list[-1] = n
                run_ends_np = np.array(run_ends_list, dtype=np.int32)

            r = len(run_ends_np)
            run_values_np = np.random.randn(r).astype(np.float64)

        # Convert to GPU tensors
        run_values_device = torch.from_numpy(run_values_np).cuda()
        run_ends_device = torch.from_numpy(run_ends_np).cuda()

        # Allocate output
        out_device = torch.zeros(n, dtype=torch.float64, device='cuda')

        # Run the kernel
        if n > 0:
            triton_module.run(run_values_device, run_ends_device, out_device, r, n)

        # Get results
        result = out_device.cpu().numpy()

        # CPU reference
        expected = cpu_run_end_decode(run_values_np, run_ends_np, n)

        # Check results
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        print(f"  PASS (r={r})")


if __name__ == "__main__":
    test_run_end_decode()
    print("All tests passed!")
