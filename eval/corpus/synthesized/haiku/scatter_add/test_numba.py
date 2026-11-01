import numpy as np
import importlib.util
from numba import cuda

spec = importlib.util.spec_from_file_location("kernel_impl", '/opgate/eval/corpus/synthesized/haiku/scatter_add/numba.py')
kernel_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel_impl)

kernel = kernel_impl.kernel
run = kernel_impl.run

def cpu_reference(idx, values, m, n):
    """CPU reference implementation"""
    out = np.zeros(n, dtype=np.float64)
    for i in range(m):
        out_idx = idx[i]
        if out_idx >= 0 and out_idx < n:
            out[out_idx] += values[i]
    return out

def test_scatter_add(m, n):
    """Test scatter_add kernel"""
    idx = np.random.randint(0, n, m, dtype=np.int32)
    values = np.random.uniform(-100, 100, m).astype(np.float64)
    out = np.zeros(n, dtype=np.float64)
    
    try:
        cuda.select_device(0)
        run(idx, values, out, m, n)
        
        ref = cpu_reference(idx, values, m, n)
        
        # Allow small numerical differences due to floating point
        if np.allclose(out, ref, rtol=1e-10, atol=1e-15):
            print(f"m={m}, n={n}: PASS")
            return True
        else:
            print(f"m={m}, n={n}: FAIL - Results don't match")
            return False
    except Exception as e:
        import traceback
        print(f"m={m}, n={n}: ERROR - {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    passed = 0
    total = 0
    
    for m in [0, 1, 1000, 100000]:
        for n in [1, 10, 100]:
            total += 1
            if test_scatter_add(m, n):
                passed += 1
    
    print(f"\nPassed {passed}/{total} tests")
    if passed == total:
        exit(0)
    else:
        exit(1)
