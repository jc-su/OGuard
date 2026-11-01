import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/tile_scan')
from numba import cuda
from numba_module import kernel, TYPES, run

def test_tile_scan():
    test_sizes = [0, 1, 10, 1024, 1025, 1000003]

    for n in test_sizes:
        print(f"Testing tile_scan with n={n}")

        # Create test data
        values = np.arange(n, dtype=np.int32)

        # Compute expected output
        TILE_SIZE = 1024
        num_tiles = (n + TILE_SIZE - 1) // TILE_SIZE
        expected_out = np.zeros(n, dtype=np.int32)
        expected_totals = np.zeros(num_tiles, dtype=np.int32)

        for i in range(n):
            tile_idx = i // TILE_SIZE
            local_idx = i % TILE_SIZE
            tile_start = tile_idx * TILE_SIZE

            # Prefix sum within tile
            prefix_sum = 0
            for j in range(tile_start, tile_start + local_idx):
                prefix_sum += values[j]
            expected_out[i] = prefix_sum

            # Tile total
            if local_idx == 0 or i == 0:
                tile_end = min(tile_start + TILE_SIZE, n)
                total = 0
                for j in range(tile_start, tile_end):
                    total += values[j]
                expected_totals[tile_idx] = total

        # Run kernel
        result_out, result_totals = run(values, n)

        # Compare
        out_match = np.array_equal(result_out, expected_out)
        totals_match = np.array_equal(result_totals, expected_totals)

        if out_match and totals_match:
            print(f"  PASS")
        else:
            print(f"  FAIL")
            if n <= 10:
                if not out_match:
                    print(f"    out result: {result_out}")
                    print(f"    out expected: {expected_out}")
                if not totals_match:
                    print(f"    totals result: {result_totals}")
                    print(f"    totals expected: {expected_totals}")

if __name__ == "__main__":
    test_tile_scan()
