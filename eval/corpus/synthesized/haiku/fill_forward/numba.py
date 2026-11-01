from numba import cuda, types

def kernel(values, valid, out, n):
    i = cuda.grid(1)
    if i < n:
        last_valid = 0
        found = False
        
        for j in range(i + 1):
            byte_idx = j // 8
            bit_idx = j % 8
            valid_bit = (valid[byte_idx] >> bit_idx) & 1
            
            if valid_bit:
                last_valid = values[j]
                found = True
        
        out[i] = last_valid if found else 0

TYPES = (types.int64[:], types.uint8[:], types.int64[:], types.int64)

def run(values, valid, out, n):
    cuda.jit(kernel)[(n + 255) // 256, 256](values, valid, out, n)
