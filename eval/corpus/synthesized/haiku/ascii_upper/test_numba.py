import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/ascii_upper')
from numba import cuda
from numba_module import kernel, TYPES, run

def test_ascii_upper():
    test_sizes = [0, 1, 10, 1000]

    for n in test_sizes:
        print(f"Testing ascii_upper with n={n}")

        # Create test strings
        strings = ["hello", "World", "TeSt123", "abc"]

        # Build offset and char arrays
        offsets = np.zeros(n + 1, dtype=np.int32)
        chars_list = []

        for i in range(n):
            offsets[i] = len(chars_list)
            s = strings[i % len(strings)]
            s_bytes = s.encode('utf-8')
            chars_list.extend(s_bytes)

        offsets[n] = len(chars_list)
        chars = np.array(chars_list, dtype=np.uint8)

        # Expected output: uppercase version
        expected_list = []
        for i in range(n):
            s = strings[i % len(strings)]
            expected_list.extend(s.upper().encode('utf-8'))
        expected = np.array(expected_list, dtype=np.uint8)

        # Run kernel
        result = run(offsets, chars, n)

        # Compare
        if np.array_equal(result, expected):
            print(f"  PASS")
        else:
            print(f"  FAIL")
            if n <= 10:
                print(f"    result: {result}")
                print(f"    expected: {expected}")

if __name__ == "__main__":
    test_ascii_upper()
