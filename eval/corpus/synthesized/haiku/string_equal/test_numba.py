import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/string_equal')
from numba import cuda
from numba_module import kernel, TYPES, run

def build_string_array(strings):
    """Build offset and char arrays from a list of strings."""
    offsets = np.zeros(len(strings) + 1, dtype=np.int32)
    chars_list = []

    offset = 0
    for s in strings:
        offsets[len(chars_list) // len(strings)] = offset
        s_bytes = s.encode('utf-8')
        chars_list.extend(s_bytes)
        offset += len(s_bytes)

    offsets[-1] = offset
    chars = np.array(chars_list, dtype=np.uint8)
    return offsets, chars

def test_string_equal():
    test_sizes = [0, 1, 10, 1000]

    for n in test_sizes:
        print(f"Testing string_equal with n={n}")

        # Create test strings
        a_strs = [f"str_a_{i}" for i in range(n)]
        b_strs = [f"str_a_{i}" if i % 2 == 0 else f"str_b_{i}" for i in range(n)]

        # Build offset and char arrays
        a_offsets = np.zeros(n + 1, dtype=np.int32)
        a_chars_list = []
        b_offsets = np.zeros(n + 1, dtype=np.int32)
        b_chars_list = []

        for i in range(n):
            a_offsets[i] = len(a_chars_list)
            a_bytes = a_strs[i].encode('utf-8')
            a_chars_list.extend(a_bytes)

            b_offsets[i] = len(b_chars_list)
            b_bytes = b_strs[i].encode('utf-8')
            b_chars_list.extend(b_bytes)

        a_offsets[n] = len(a_chars_list)
        b_offsets[n] = len(b_chars_list)

        a_chars = np.array(a_chars_list, dtype=np.uint8)
        b_chars = np.array(b_chars_list, dtype=np.uint8)

        # Expected result
        expected = np.array([1 if a_strs[i] == b_strs[i] else 0 for i in range(n)], dtype=np.int8)

        # Run kernel
        result = run(a_offsets, a_chars, b_offsets, b_chars, n)

        # Compare
        if np.array_equal(result, expected):
            print(f"  PASS")
        else:
            print(f"  FAIL")
            if n <= 10:
                print(f"    result: {result}")
                print(f"    expected: {expected}")

if __name__ == "__main__":
    test_string_equal()
