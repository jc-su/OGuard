import numpy as np
import importlib.util
from numba import cuda

spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/diff/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(values, n):
    """CPU reference implementation"""
    out = np.zeros(n, dtype=np.int64)
    if n > 0:
        out[0] = values[0]
        for i in range(1, n):
            out[i] = values[i] - values[i - 1]
    return out

def test_diff(n):
    """Test diff kernel"""
    values = np.random.randint(-1000, 1000, n, dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    
    try:
        cuda.select_device(0)
        run(values, out, n)
        
        ref = cpu_reference(values, n)
        
        if np.array_equal(out, ref):
            print(f"n={n}: PASS")
            return True
        else:
            print(f"n={n}: FAIL - Results don't match")
            return False
    except Exception as e:
        import traceback
        print(f"n={n}: ERROR - {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    passed = 0
    total = 0
    
    for n in [0, 1, 1000, 1000003]:
        total += 1
        if test_diff(n):
            passed += 1
    
    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
