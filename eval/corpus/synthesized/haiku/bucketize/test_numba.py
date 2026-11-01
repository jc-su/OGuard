import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/bucketize')
from numba import cuda
from numba_module import kernel, TYPES, run

def test_bucketize():
    test_configs = [(0, 5), (1, 5), (10, 3), (1000, 10)]

    for n, b in test_configs:
        print(f"Testing bucketize with n={n}, b={b}")

        # Create values and bounds
        values = np.random.rand(n).astype(np.float64) * 100
        bounds = np.array([10.0, 25.0, 50.0, 75.0, 90.0][:b], dtype=np.float64)
        bounds = np.sort(bounds)

        # Expected output
        expected = np.zeros(n, dtype=np.int32)
        for i in range(n):
            count = 0
            for j in range(b):
                if bounds[j] <= values[i]:
                    count += 1
                else:
                    break
            expected[i] = count

        # Run kernel
        result = run(values, bounds, n, b)

        # Compare
        if np.array_equal(result, expected):
            print(f"  PASS")
        else:
            print(f"  FAIL")
            if n <= 10:
                print(f"    result: {result}")
                print(f"    expected: {expected}")
                print(f"    values: {values}")
                print(f"    bounds: {bounds}")

if __name__ == "__main__":
    test_bucketize()
