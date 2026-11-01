import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/lag')
from numba import cuda
from numba_module import kernel, TYPES, run

def test_lag():
    test_sizes = [0, 1, 10, 1000]
    k = 2

    for n in test_sizes:
        print(f"Testing lag with n={n}, k={k}")

        # Create values
        values = np.arange(n, dtype=np.float64)

        # Create partitions: first half in partition 0, second half in partition 1
        first = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if i < n // 2:
                first[i] = 0
            else:
                first[i] = n // 2

        # Expected output
        expected = np.zeros(n, dtype=np.float64)
        for i in range(n):
            lag_idx = i - k
            first_row = first[i]

            if lag_idx >= first_row:
                expected[i] = values[lag_idx]
            else:
                expected[i] = float('nan')

        # Run kernel
        result = run(values, first, n, k)

        # Compare
        match = True
        for i in range(n):
            if np.isnan(expected[i]):
                if not np.isnan(result[i]):
                    match = False
                    break
            elif np.abs(result[i] - expected[i]) > 1e-9:
                match = False
                break

        if match:
            print(f"  PASS")
        else:
            print(f"  FAIL")
            if n <= 10:
                print(f"    result: {result}")
                print(f"    expected: {expected}")

if __name__ == "__main__":
    test_lag()
