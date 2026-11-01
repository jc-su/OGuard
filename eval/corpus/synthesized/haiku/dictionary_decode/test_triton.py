import torch
import numpy as np
import importlib.util


def load_module(path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location("kernel_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_dictionary_decode(codes, dictionary):
    """CPU reference implementation."""
    return dictionary[codes]


def test_dictionary_decode():
    """Test dictionary_decode kernel on various input sizes."""
    triton_module = load_module("/scratch/heldout/haiku/dictionary_decode/triton.py")

    test_sizes = [0, 1, 1000, 1000003]

    for n in test_sizes:
        print(f"Testing dictionary_decode with n={n}")

        # Create random codes (in range [0, 256))
        d = 256
        codes_np = np.random.randint(0, d, size=n, dtype=np.int32)
        codes_device = torch.from_numpy(codes_np).cuda()

        # Create random dictionary
        dict_np = np.random.randn(d).astype(np.float64)
        dict_device = torch.from_numpy(dict_np).cuda()

        # Allocate output
        out_device = torch.zeros(n, dtype=torch.float64, device='cuda')

        # Run the kernel
        triton_module.run(codes_device, dict_device, out_device, n, d)

        # Get results
        result = out_device.cpu().numpy()

        # CPU reference
        expected = cpu_dictionary_decode(codes_np, dict_np)

        # Check results
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        print(f"  PASS")


if __name__ == "__main__":
    test_dictionary_decode()
    print("All tests passed!")
