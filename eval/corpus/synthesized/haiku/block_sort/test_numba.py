import numpy as np
import sys
sys.path.insert(0, '/opgate/eval/corpus/synthesized/haiku/block_sort')
from numba import cuda
from numba_module import kernel, TYPES, run

def test_block_sort():
    test_sizes = [0, 1, 10, 1024, 1025, 1000003]

    for n in test_sizes:
        print(f"Testing block_sort with n={n}")

        # Create test data with random values
        np.random.seed(42)
        keys = np.random.randint(0, 1000, n, dtype=np.int32)

        # Compute expected output
        TILE_SIZE = 1024
        expected = np.copy(keys)

        for tile_idx in range((n + TILE_SIZE - 1) // TILE_SIZE):
            tile_start = tile_idx * TILE_SIZE
            tile_end = min(tile_start + TILE_SIZE, n)
            expected[tile_start:tile_end] = np.sort(expected[tile_start:tile_end])

        # Run kernel
        result = run(keys, n)

        # Compare
        if np.array_equal(result, expected):
            print(f"  PASS")
        else:
            print(f"  FAIL")
            if n <= 10:
                print(f"    result: {result}")
                print(f"    expected: {expected}")
                # Check first few tiles
                for tile_idx in range(min(2, (n + TILE_SIZE - 1) // TILE_SIZE)):
                    tile_start = tile_idx * TILE_SIZE
                    tile_end = min(tile_start + TILE_SIZE, n)
                    print(f"    tile {tile_idx}: result={result[tile_start:tile_end]}, expected={expected[tile_start:tile_end]}")

if __name__ == "__main__":
    test_block_sort()
