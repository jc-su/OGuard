import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/coalesce')
from numba import cuda
from numba_module import kernel, TYPES, run

def make_valid_bitmap(n, valid_indices):
    """Create a validity bitmap where valid_indices are set to 1."""
    bitmap_size = (n + 7) // 8
    bitmap = np.zeros(bitmap_size, dtype=np.uint8)
    for idx in valid_indices:
        byte_idx = idx // 8
        bit_idx = idx % 8
        bitmap[byte_idx] |= (1 << bit_idx)
    return bitmap

def test_coalesce():
    test_sizes = [0, 1, 1000, 1000003]

    for n in test_sizes:
        print(f"Testing coalesce with n={n}")

        a = np.random.rand(n).astype(np.float64)
        b = np.random.rand(n).astype(np.float64)

        # Create bitmap with all rows valid
        valid_indices = list(range(n))
        valid = make_valid_bitmap(n, valid_indices)

        # Expected output: out[i] = a[i] (all valid)
        expected = a.copy()

        # Run kernel
        result = run(a, b, n)

        # Compare
        if np.allclose(result, expected, equal_nan=True):
            print(f"  PASS")
        else:
            print(f"  FAIL: result != expected")
            if n <= 10:
                print(f"    result: {result}")
                print(f"    expected: {expected}")

if __name__ == "__main__":
    test_coalesce()
