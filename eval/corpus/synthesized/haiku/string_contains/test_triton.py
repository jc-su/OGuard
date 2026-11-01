import torch
import numpy as np
import importlib.util


def load_module(path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location("kernel_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_string_contains(offsets, chars, pattern):
    """CPU reference implementation."""
    n = max(0, len(offsets) - 1)
    match = np.zeros(n, dtype=np.int8)

    for i in range(n):
        start = offsets[i]
        end = offsets[i + 1]
        string_bytes = chars[start:end]

        # Check if pattern is in string
        if len(pattern) == 0:
            match[i] = 1  # Empty pattern occurs in every string
        elif len(pattern) <= len(string_bytes):
            # Use simple substring search
            found = False
            for j in range(len(string_bytes) - len(pattern) + 1):
                if np.array_equal(string_bytes[j:j+len(pattern)], pattern):
                    found = True
                    break
            match[i] = 1 if found else 0
        else:
            match[i] = 0

    return match


def test_string_contains():
    """Test string_contains kernel on various input sizes."""
    triton_module = load_module("/scratch/heldout/haiku/string_contains/triton.py")

    test_cases = [
        # (n, strings_list, pattern)
        (0, [], np.array([], dtype=np.uint8)),  # Empty
        (1, ["hello"], np.array([ord('l')], dtype=np.uint8)),  # Single string
        (1, ["hello"], np.array([], dtype=np.uint8)),  # Empty pattern
        (5, ["hello", "world", "test", "abc", "xyz"], np.array([ord('o')], dtype=np.uint8)),  # Simple pattern
    ]

    for n, strings, pattern in test_cases:
        print(f"Testing string_contains with n={n}, pattern_len={len(pattern)}")

        # Create offsets and chars from strings (always use at least offset 0)
        offsets_list = [0]
        chars_list = []
        for s in strings:
            if isinstance(s, str):
                chars_list.extend(s.encode())
            else:
                chars_list.extend(s)
            offsets_list.append(len(chars_list))

        offsets_np = np.array(offsets_list, dtype=np.int32)
        chars_np = np.array(chars_list, dtype=np.uint8)
        offsets_device = torch.from_numpy(offsets_np).cuda()
        chars_device = torch.from_numpy(chars_np).cuda()

        pattern_device = torch.from_numpy(pattern).cuda()
        match_device = torch.zeros(n, dtype=torch.int8, device='cuda')

        # Run the kernel
        if n > 0:
            triton_module.run(offsets_device, chars_device, pattern_device, match_device, n, len(pattern))

        # Get results
        result = match_device.cpu().numpy()

        # CPU reference
        expected = cpu_string_contains(offsets_np, chars_np, pattern)

        # Check results
        np.testing.assert_array_equal(result, expected)
        print(f"  PASS")


def test_string_contains_large():
    """Test with larger random data."""
    triton_module = load_module("/scratch/heldout/haiku/string_contains/triton.py")

    for n in [1000, 10000]:
        print(f"Testing string_contains with n={n} (random strings)")

        # Generate random strings
        offsets_list = [0]
        chars_list = []
        for _ in range(n):
            str_len = np.random.randint(1, 100)
            for _ in range(str_len):
                chars_list.append(np.random.randint(0, 256, dtype=np.uint8))
            offsets_list.append(len(chars_list))

        offsets_np = np.array(offsets_list, dtype=np.int32)
        chars_np = np.array(chars_list, dtype=np.uint8)

        # Random pattern (short)
        pattern = np.random.randint(0, 256, size=5, dtype=np.uint8)

        # GPU execution
        offsets_device = torch.from_numpy(offsets_np).cuda()
        chars_device = torch.from_numpy(chars_np).cuda()
        pattern_device = torch.from_numpy(pattern).cuda()
        match_device = torch.zeros(n, dtype=torch.int8, device='cuda')

        triton_module.run(offsets_device, chars_device, pattern_device, match_device, n, len(pattern))
        result = match_device.cpu().numpy()

        # CPU reference
        expected = cpu_string_contains(offsets_np, chars_np, pattern)

        # Check results
        np.testing.assert_array_equal(result, expected)
        print(f"  PASS")


if __name__ == "__main__":
    test_string_contains()
    test_string_contains_large()
    print("All tests passed!")
