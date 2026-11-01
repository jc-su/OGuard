import numpy as np
import numba
from numba import cuda as numba_cuda

# Import from numba.py
from numba import cuda
exec(open('numba.py').read())

test_sizes = [0, 1, 1000, 1000003]

for n in test_sizes:
    try:
        valid_len = (n + 7) // 8
        out_len = (n + 7) // 8
        
        h_values = np.arange(n, dtype=np.int64) - n // 2
        h_valid = np.zeros(valid_len, dtype=np.uint8)
        h_out = np.zeros(out_len, dtype=np.uint8)
        
        # Set some validity bits
        for i in range(0, n, 3):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < valid_len:
                h_valid[byte_idx] |= (1 << bit_idx)
        
        d_values = cuda.to_device(h_values)
        d_valid = cuda.to_device(h_valid)
        d_out = cuda.to_device(h_out)
        
        run(d_values, d_valid, d_out, n, -100, 100)
        
        print(f"n={n}: PASS")
    except Exception as e:
        print(f"n={n}: FAIL - {str(e)[:60]}")
