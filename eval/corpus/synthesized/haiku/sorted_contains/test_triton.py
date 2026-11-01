import torch
import numpy as np
import importlib.util


def load_module(path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location("kernel_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_sorted_contains(keys, members):
    """CPU reference implementation."""
    # Use set for O(1) lookup
    members_set = set(members)
    return np.array([1 if k in members_set else 0 for k in keys], dtype=np.int8)


def test_sorted_contains():
    """Test sorted_contains kernel on various input sizes."""
    triton_module = load_module("/scratch/heldout/haiku/sorted_contains/triton.py")

    test_cases = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (10, 5),
        (1000, 100),
        (1000003, 10000),
    ]

    for n, s in test_cases:
        print(f"Testing sorted_contains with n={n}, s={s}")

        # Create keys and members
        if n == 0:
            keys_np = np.array([], dtype=np.int64)
        else:
            keys_np = np.random.randint(-1000, 1000, size=n, dtype=np.int64)

        if s == 0:
            members_np = np.array([], dtype=np.int64)
        else:
            members_np = np.sort(np.random.choice(range(-1000000, 1000000), size=s, replace=False).astype(np.int64))

        # GPU execution
        keys_device = torch.from_numpy(keys_np).cuda()
        members_device = torch.from_numpy(members_np).cuda()
        match_device = torch.zeros(n, dtype=torch.int8, device='cuda')

        # Run the kernel
        if n > 0:
            triton_module.run(keys_device, members_device, match_device, n, s)

        # Get results
        result = match_device.cpu().numpy()

        # CPU reference
        expected = cpu_sorted_contains(keys_np, members_np)

        # Check results
        np.testing.assert_array_equal(result, expected)
        print(f"  PASS")


if __name__ == "__main__":
    test_sorted_contains()
    print("All tests passed!")
