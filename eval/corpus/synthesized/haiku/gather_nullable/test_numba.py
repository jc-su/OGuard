import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/gather_nullable')
from numba import cuda
from numba_module import kernel, TYPES, run

def test_gather_nullable():
    test_configs = [(0, 10), (1, 10), (1000, 1000), (1000003, 1000003)]

    for m, n in test_configs:
        print(f"Testing gather_nullable with m={m}, n={n}")

        # Create input column
        col = np.arange(n, dtype=np.float64)

        # Create indices: mix of valid and -1
        idx = np.zeros(m, dtype=np.int32)
        for i in range(m):
            if i % 3 == 0 and i > 0:
                idx[i] = -1  # Mark as null
            else:
                idx[i] = i % n  # Valid index

        # Expected output
        expected = np.zeros(m, dtype=np.float64)
        for i in range(m):
            if idx[i] == -1:
                expected[i] = float('nan')
            else:
                expected[i] = col[idx[i]]

        # Run kernel
        result = run(col, idx, m, n)

        # Compare
        match = True
        for i in range(m):
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
            if m <= 10:
                print(f"    result: {result}")
                print(f"    expected: {expected}")

if __name__ == "__main__":
    test_gather_nullable()
