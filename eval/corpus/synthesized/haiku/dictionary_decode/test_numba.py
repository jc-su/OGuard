import numpy as np
from numba import cuda, types

def kernel(codes, dictionary, out, n, d):
    """Decode dictionary-encoded column."""
    idx = cuda.grid(1)
    if idx < n:
        code = codes[idx]
        out[idx] = dictionary[code]

TYPES = (types.int32[:], types.float64[:], types.float64[:], types.int64, types.int64)

def run(codes_np, dictionary_np, out_np, n, d):
    """Launch kernel on device arrays."""
    codes = cuda.to_device(codes_np)
    dictionary = cuda.to_device(dictionary_np)
    out = cuda.to_device(out_np)

    threadsperblock = 256
    blockspergrid = max(1, (n + threadsperblock - 1) // threadsperblock)
    kernel_fn = cuda.jit(kernel)
    kernel_fn[blockspergrid, threadsperblock](codes, dictionary, out, n, d)

    return out.copy_to_host()

def cpu_reference(codes_np, dictionary_np, n):
    """Compute reference on CPU."""
    return dictionary_np[codes_np[:n]]

def test_dictionary_decode():
    """Test dictionary_decode kernel."""
    for n in [0, 1, 1000, 1000003]:
        d = min(256, max(1, n // 10 + 1))

        # Create test data
        codes = np.random.randint(0, d, size=n, dtype=np.int32)
        dictionary = np.random.randn(d).astype(np.float64)
        out = np.zeros(n, dtype=np.float64)

        # Run kernel
        out = run(codes, dictionary, out, n, d)

        # Get reference
        ref = cpu_reference(codes, dictionary, n)

        # Check results
        np.testing.assert_allclose(out, ref, rtol=1e-10)

        print(f"✓ n={n}, d={d}: passed")

if __name__ == "__main__":
    test_dictionary_decode()
    print("All tests passed!")
