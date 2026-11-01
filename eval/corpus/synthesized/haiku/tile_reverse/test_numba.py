import numpy as np
import importlib.util
from numba import cuda

spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/tile_reverse/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

TILE_SIZE = 1024

def cpu_reference(values, n):
    """CPU reference implementation"""
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        tile_id = i // TILE_SIZE
        pos_in_tile = i % TILE_SIZE
        tile_start = tile_id * TILE_SIZE
        tile_end = min(tile_start + TILE_SIZE, n)
        tile_len = tile_end - tile_start
        rev_pos = tile_len - 1 - pos_in_tile
        out[i] = values[tile_start + rev_pos]
    return out

def test_tile_reverse(n):
    """Test tile_reverse kernel"""
    values = np.random.randint(-1000, 1000, n, dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    
    try:
        cuda.select_device(0)
        run(values, out, n)
        
        ref = cpu_reference(values, n)
        
        if np.array_equal(out, ref):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            if n <= 2000:
                print(f"Expected: {ref}")
                print(f"Got: {out}")
            return False
    except Exception as e:
        import traceback
        print(f"n={n}: ERROR - {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    passed = 0
    total = 0
    
    for n in [0, 1, 1024, 1025, 2048, 1000003]:
        total += 1
        if test_tile_reverse(n):
            passed += 1
    
    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
