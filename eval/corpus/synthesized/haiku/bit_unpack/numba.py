from numba import cuda, types

def kernel(packed, out, n, bits):
    i = cuda.grid(1)
    if i < n:
        bit_pos = i * bits
        word_idx = bit_pos // 32
        bit_idx = bit_pos % 32
        
        code = 0
        for b in range(bits):
            pos = bit_pos + b
            w_idx = pos // 32
            b_idx = pos % 32
            bit = (packed[w_idx] >> b_idx) & 1
            code |= (bit << b)
        
        out[i] = code

TYPES = (types.uint32[:], types.uint32[:], types.int64, types.int64)

def run(packed, out, n, bits):
    cuda.jit(kernel)[(n + 255) // 256, 256](packed, out, n, bits)
